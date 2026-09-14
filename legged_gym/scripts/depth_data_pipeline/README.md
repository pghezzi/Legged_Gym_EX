# Terrain classifier training

`train_terrain_classifiers.py` is the recommended entry point for training,
hyperparameter search, structural-test evaluation, Bayesian-filter fitting, and
ordered-test evaluation. It delegates to the existing approach-specific scripts
and places each run in a separate output directory.

## Expected dataset layout

The simplest layout is:

```text
dataset_root/
├── structural/
│   ├── train.pt
│   ├── val.pt
│   ├── calibration.pt
│   └── test.pt
└── bayesian/
    ├── train.pt          # optional, preferred for transition fitting
    ├── calibration.pt    # optional fallback for transition fitting
    ├── val.pt
    └── test.pt
```

The structural files train and evaluate the instantaneous classifier. Bayesian
files contain ordered sequences. If Bayesian `train.pt` or `calibration.pt` is not
available, transition fitting reuses `val.pt` and emits an optimism warning.

Alternate subfolder names `classifier/`, `bayes/`, and `sequences/` are detected.
For other layouts, pass `--classifier-data` and `--bayesian-data` explicitly.

## Training

Run both NN architectures (each produces deterministic and MC-Dropout deployments):

```bash
python -m legged_gym.scripts.depth_data_pipeline.train_terrain_classifiers \
  --dataset /path/to/dataset_root \
  --approach all
```

Run selected approaches and choose an output directory:

```bash
python -m legged_gym.scripts.depth_data_pipeline.train_terrain_classifiers \
  --classifier-data /data/terrain/structural \
  --bayesian-data /data/terrain/bayesian \
  --output /results/terrain_suite \
  --approach feature_nn raw_depth_nn \
  --batch-size 128
```

Approach names are:

- `feature_nn`
- `raw_depth_nn`
- `all`

Batch/chunk processing is enabled by default. `--batch-size` controls feature
extraction, classifier-score inference, and NN batches. To process each split as
a single batch, use:

```bash
python -m legged_gym.scripts.depth_data_pipeline.train_terrain_classifiers \
  --dataset /path/to/dataset_root \
  --approach feature_nn \
  --no-batch-processing
```

Full-split processing can require substantially more RAM and VRAM. Sequential
search caches only compact CPU classifier scores/probabilities, not images or
engineered-feature tensors.

If `--output` is omitted, a timestamped suite directory is created under
`depth_waq_selector/full_models/`. If `--dataset` is omitted, the script checks
`$TERRAIN_CLASSIFIER_DATASET`, then `depth_waq_selector/processed_data/`.

Each approach directory contains its classifier/filter artifacts, searches,
parameters, metrics, and standardized `results.json`. The suite root contains
`suite_manifest.json` and, by default, comparison CSV/JSON files. Use
`--skip-comparison` to omit automatic comparison or `--continue-on-error` to run
remaining approaches after one fails.

Training uses the frozen paper-search configurations (no NN hyperparameter search):
feature deterministic `dropout_p=0, weight_decay=1e-5`, feature MC
`dropout_p=0.10, weight_decay=1e-4`, and raw-depth deterministic/MC
`dropout_p=0.20, weight_decay=1e-5`. Models train for at most 50 epochs with
validation-loss early stopping and best-weight rollback, then independently cache
and time batched MC10/25/50 logits. Deterministic and MC branches remain separate.
The raw-depth network concatenates robot roll, pitch, and roll/pitch/yaw angular
velocities with the flattened CNN representation before its hidden FC layers.
The ordered-validation search uses structured score/low-delay/transition/false-event
frontiers across fixed persistence, candidate-directed release, MI gating, refined
ambiguity handling, adaptive beta, and accumulated transition evidence. It also
records Pareto frontiers and a frozen controlled C0-C3 ablation lineage. For each
retained classifier, an independent EMA-score + patience
baseline searches `ema_alpha=0.40/0.60/0.80/1.0` and `patience=1/2`. Results report
the validation/CV-selected EMA baseline beside the best Bayes filter; ordered test
data remains reporting-only. `deployment_deterministic.json` and
`deployment_mc.json` contain everything required for automatic deployment.

MC deployments additionally compare the protected temporal baseline frontier with
MI gating, MI-adaptive observation strength, and uncertainty-weighted accumulated
transition evidence. Candidate agreement is diagnostic only and never gates a
transition. Search/development Experiment A-C CSV/JSON files and self-contained
selected configuration files are written for later no-search paper evaluation.
Selected and stage-frontier configurations also write compact `.pt` per-frame
traces under `temporal_traces/`. Use `load_temporal_trace()` and
`recompute_temporal_trace_metrics()` from `sequential_terrain_filter_extensions`
to change transition-window radii or build offline failure/uncertainty plots
without rerunning neural inference or filter search.

## Comparing saved results

Comparison never reruns training. Compare a completed suite:

```bash
python -m legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results \
  /results/terrain_suite \
  --output /results/terrain_suite/comparison
```

Or provide individual run directories/files:

```bash
python -m legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results \
  /results/feature_nn /results/raw_nn \
  --output terrain_comparison
```

The command writes stage-level and selected-deployment comparisons:

- `<output>_winners.csv`
- `<output>_stages.csv`
- `<output>.json`

The comparison directory also includes `stage_frontiers`, `pareto_frontiers`,
Experiment A-C search/selection files, and best-overall/low-delay deployment JSON.

The suite also writes `suite_deployments.json` identifying the four learned
deployments and the global validation/CV-selected deterministic, MC, and overall
winners.

Outputs include structural and ordered instantaneous metrics, uncertainty and
calibration summaries, temporal validation/CV scores, ordered-test metrics, and
the selected Bayes and EMA baseline parameters.

## Frozen offline paper Experiments 1--2

### Docker launcher (persistent shared outputs)

From the repository root, use Bash (no TTY required). This reuses the
`leggedgym-ex:isaacgym` image/venv and mounts source subdirectories, not the
workspace containing `.venv`. Inputs are mounted read-only. Every invocation
creates a **new** run; an existing `--run-id` is rejected to protect old results.

```bash
# Offline training/evaluation only; already-compiled datasets required.
bash legged_gym/scripts/run_paper_experiments_docker.sh offline \
  --classifier-data /data/compiled/structural --ordered-data /data/compiled/ordered \
  --output-root /data/paper_runs --run-id offline_v1 --gpu 0

# Locomotion using a completed offline run (no classifier retraining).
bash legged_gym/scripts/run_paper_experiments_docker.sh locomotion \
  --paper-offline-dir /data/paper_runs/offline_v1/offline \
  --jit /models/specialists.pt --distilled-jit /models/distilled.pt \
  --output-root /data/paper_runs --run-id locomotion_v1

# Offline first, then headless locomotion using its new classifiers.
bash legged_gym/scripts/run_paper_experiments_docker.sh all \
  --classifier-data /data/compiled/structural --ordered-data /data/compiled/ordered \
  --jit /models/specialists.pt --distilled-jit /models/distilled.pt \
  --output-root /data/paper_runs --gpu 1

# Regenerate PNG/PDFs from the self-contained bundle, without datasets/models.
bash legged_gym/scripts/run_paper_experiments_docker.sh plot-only \
  --bundle /data/paper_runs/offline_v1/offline/figure_data.pt \
  --output-root /data/paper_runs --gpu none
```

Use `--image` to override the image, `--dry-run` to validate paths/show commands
without creating outputs, and `--help` for all options. Separate passthrough uses
one token per option, e.g. `--offline-arg --batch-size --offline-arg 128` or
`--locomotion-arg --eval-seeds --locomotion-arg 101 --locomotion-arg 202`.
Path/mode overrides must use the wrapper's options; no shell command strings
are evaluated. Without passthrough, existing experimental defaults are unchanged.
Host GPU indices are exposed as `cuda:0` inside the single-GPU container.

The launcher prints stage start/end and a heartbeat every 30 seconds showing
elapsed time and time since the last console output. Set `--progress-interval 10`
for more frequent updates, or `0` to disable heartbeats. These messages are saved
in `logs/progress.log`; actual experiment messages remain in `logs/offline.log`
(or the relevant stage log). A heartbeat means the launcher is still waiting,
not proof that computation is advancing. Offline phase/model/seed messages and
figure counts provide actual progress; figures report every 25 PNG/PDF pairs.
An already-running Python process will not pick up these changes automatically.

All checkpoints, metrics, figures/bundle and replay data persist under
`<output-root>/<run-id>/offline/` and `locomotion/`. Console/simulator logs live
in `logs/`; `commands.sh`, `mounts.txt`, `run_metadata.txt`, `git_status.txt`, and
per-stage/overall `exit_status` files record execution. Container paths are
`/paper/offline`, `/paper/locomotion`, `/paper/logs` (also the workspace `logs/`).
Mappings in `mounts.txt` resolve these paths to the host when moving artifacts.

The launcher uses `umask 000` and normalizes only the **new managed run** to
0777 directories / 0666 files on exit, including failure. Everyone can modify
these results: use trusted storage with traversable ancestor directories.
Existing results, inputs, repository permissions and the host environment are
untouched. Host crashes/SIGKILL or inaccessible Docker may prevent final cleanup;
cleanup failures are reported. A lightweight Docker-only write/failure check is
available with `--smoke-test success` or `--smoke-test failure` (exit 23); combine
with `plot-only --bundle <any-readable-file> --gpu none`. It never runs an
experiment. Automated checks: `RUN_DOCKER_SMOKE=1 python -m unittest discover
-s tests -p test_paper_experiments_docker.py` (uses local `ubuntu:20.04`, overridable
with `PAPER_SMOKE_IMAGE`). No full sweep is needed to test the launcher.

### Automatic training-cost audit and relocated inputs

`offline`, `locomotion`, and `all` now automatically aggregate post-specialist
training costs into the new run's `costs/` directory. `plot-only` does so only
when `--paper-offline-dir` is also supplied. No evaluation runtime is used as
training cost. Missing collection/compilation/training records leave totals
unavailable, rather than zero. A locomotion failure still permits the independent
cost audit when offline results exist, while preserving the evaluation exit code.

Add these repeatable options to any applicable launch:

```bash
  --collection-cost /data/capture.pt.training_cost.json \
  --compilation-cost /data/compiled/training_cost.json \
  --distillation-run /models/distill_seed_0 \
  --distillation-run /models/distill_seed_1 \
  --distillation-run /models/distill_seed_2 \
  --deployment-artifact distilled:0 /models/student_seed_0.pt
```

Aggregate existing runs without training or evaluation:

```bash
bash legged_gym/scripts/run_paper_experiments_docker.sh cost-only \
  --paper-offline-dir /data/paper_runs/offline_v1/offline --gpu none \
  --collection-cost /data/capture.pt.training_cost.json \
  --compilation-cost /data/compiled/training_cost.json \
  --path-map /inputs/classifier /data/compiled \
  --distillation-run /models/distill_seed_0 \
  --deployment-artifact distilled:0 /models/student_seed_0.pt \
  --output-root /data/paper_runs --run-id cost_audit_v1
```

All additional inputs are read-only. `--path-map RECORDED_PREFIX HOST_PATH`
mounts the current host file/directory and resolves references recorded under an
old host/container prefix; repeat for other relocated roots. Longest matching
prefix wins. Current host mounts and the previous `/paper/offline` location are
mapped automatically. Original manifests/sidecars are never edited. Resolved
identities deduplicate sources shared by training/calibration; unreferenced
ordered-test collection is not charged. Mappings are recorded in commands and
`costs/training_cost_manifest.json`. Raw image files are not needed if their
linked sidecars/provenance are supplied. Missing optional data stays unavailable.

Deployment accounting adds `deployment_size_mb` (MiB) without replacing the
historical `artifact_size_mb` training-checkpoint field. Router sizes include
the frozen loader's manifest, classifier/model arguments, and feature
extractor/standardizer when applicable. Training optimizer checkpoints and
pre-existing specialists are not added. Explicit exports can be supplied with
`--deployment-artifact feature_nn[:SEED]|raw_depth_nn[:SEED]|distilled[:SEED] FILE`;
repeat for split exports. Unscoped files are explicitly shared by that method's
runs; use `:0`, `:1`, `:2` for independent seed exports. Locomotion's
`--distilled-jit` is the shared export reference unless explicit distilled
artifacts are provided. Missing deployment components make their total
unavailable. Files are counted once per deployment, not summed across alternatives.

Cost CSV/JSON, LaTeX, and PNG/PDF outputs retain the same universal permissions.
See `logs/costs.log`, `logs/costs.exit_status`, and `commands.sh`. To test real
cost-only aggregation on synthetic relocated sidecars (no experiments):
`RUN_DOCKER_COSTS=1 python -m unittest discover -s tests -p test_paper_experiments_docker.py`.

To train exactly three deterministic seeded copies of the fixed feature/raw-depth
NNs and evaluate instantaneous, fixed EMA, and fixed persistent-Bayes results
without running any search:

```bash
python -m legged_gym.scripts.depth_data_pipeline.evaluate_paper_offline_experiments_1_2 \
  --dataset /path/to/leakage_safe_compiled_dataset \
  --output paper_offline_eval
```

Use `--classifier-data` and `--ordered-data` for nonstandard structural/ordered
folder layouts. The output contains per-seed and mean/std CSV/JSON metrics,
checkpoints and preprocessing artifacts, a reproducibility manifest, and the
initial Experiment 1--2 figures.

## Post-specialist training-cost audit

New classifier collection runs, frozen offline classifier runs, and multi-skill
distillation runs write training-cost JSON sidecars without changing their
training behavior. Aggregate three classifier seeds and up to three distillation
seeds with:

```bash
python -m legged_gym.scripts.depth_data_pipeline.summarize_training_costs \
  --paper-offline-dir paper_offline_eval \
  --collection-cost /path/to/capture.pt.training_cost.json \
  --distillation-run /path/to/distill_seed_0 /path/to/distill_seed_1 /path/to/distill_seed_2 \
  --output training_cost_audit
```

The report uses the common boundary from frozen specialist policies to a
deployable router or unified student. Every cost carries measured,
reconstructed, or unavailable provenance; teacher-labelled distillation
rollouts are counted once even though they serve both data generation and
student-policy training.
