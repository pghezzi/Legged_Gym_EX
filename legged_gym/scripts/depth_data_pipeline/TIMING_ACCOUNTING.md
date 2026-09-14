# Post-specialist timing boundaries

New cost records carry `timing_schema_version: 2`; existing field names remain.
All durations use `perf_counter`. GPU stage boundaries synchronize the selected
CUDA device using `WallTimer`/`synchronize`. Import/startup before the main entry
function and writing the accounting sidecar itself are outside the boundary.

The six **non-overlapping, additive** fields are:

| Field | Boundary |
|---|---|
| `setup_s` | Configuration, input loading, environment/model construction, and preparation before the measured work starts |
| `data_generation_s` | Collector interaction loop (including online capture/filtering), or distillation rollouts and teacher labeling |
| `preprocessing_s` | Collector post-rollout stacking/filtering; compiler sampling/splitting/merging; classifier calibration fitting, feature extraction/standardization or raw input packing |
| `optimization_s` | Classifier fit, including validation/early stopping needed for fitting; or distillation update calls |
| `artifact_serialization_s` | Dataset/model/feature artifact writes, including periodic distillation checkpoint writes |
| `overhead_s` | Remaining accounted elapsed time, including inter-stage bookkeeping/logging |

Their sum equals `total_wallclock_s`. Non-applicable stages on new records are
explicit zeroes; missing historical measurements are `null`/`unavailable`.
`gpu_hours = total_wallclock_s * gpu_count / 3600` charges allocated-device
wall-clock, including CPU work and I/O within the boundary, **not GPU-active time**.
`gpu_active_time_s` remains unavailable. The existing device helper describes one
selected CUDA device (or zero for CPU); external reservations/multiple-process
allocations require separate allocation metadata.

Collection starts before environment and specialist construction and ends after
dataset serialization. Distillation starts before construction through `train.py`
and ends after the final checkpoint. Calling its runner directly excludes the
already-created environment. Checkpoint loading and resume behavior are unchanged.

Compilation writes `training_cost.json` next to `split_manifest.json` and the
compiled tensors. `data_loading_s` is a subdivision of setup; `sampling_s`,
`splitting_s`, and `merging_s` subdivide preprocessing. These fields are **not**
added again to the six-stage sum. Compilation runs on CPU, with GPU count zero.

Classifier accounting includes train/validation data loading, model construction,
training preprocessing, fitting, and artifact serialization. Calibration loading
is included in feature preprocessing. Structural/ordered test loading, sequence
ID preparation, inference, metrics, and evaluation output writing are excluded.
`training_runtime_seconds` still describes fit only; `total_classifier_training_s`
now covers the complete classifier integration boundary. Shared preparation is
measured once and charged once to each independently reported seed/architecture.

The summarizer follows `structural_dataset`/`structural_split_manifest` provenance,
including calibration sources, to locate collection and compilation sidecars.
It deduplicates resolved dataset paths within each alternative and excludes the
ordered evaluation dataset. `--collection-cost` and `--compilation-cost` can supply
sidecars at other locations; their `dataset_path` must identify the same artifact.
Paths recorded on another host may need relocation. Missing referenced sidecars
produce unavailable totals rather than partial totals that appear complete.

`compilation_s` is a **subtotal**, already distributed across the additive stages;
it must not be added again. Tables include it, and stacked plots show the six
stages. An incomplete breakdown is labeled unavailable instead of plotting missing
stages as zero. Historical sidecars with incompatible collection/classifier
boundaries retain their source values but cannot establish complete router totals.

Compilation is charged at artifact granularity, including jointly prepared test
splits and calibration artifacts, even when an alternative does not consume every
split. No per-split time allocation is inferred. GPU utilization, data transfers
inside fit/rollout, and individual teacher loading times are not profiled separately.
