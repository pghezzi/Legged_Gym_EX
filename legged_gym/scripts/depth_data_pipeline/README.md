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
