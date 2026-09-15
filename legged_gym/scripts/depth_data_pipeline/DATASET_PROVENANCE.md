# Observation and sequence provenance

New raw captures contain a `provenance` dictionary with a UUID `capture_id` /
`source_id`, and aligned `[T,N]` tensors: `original_env_ids`, `episode_ids`,
`control_step_indices`, and `frame_indices`. Control steps start at 1 and frame
indices at 0. The episode counter observes every `env.reset_idx` call, including
non-camera ticks and manual resets after a capture. It is snapshotted with the
post-step observation actually saved, before subsequent manual resets. These
indices describe observation delivery, not the optical exposure time of delayed
depth pixels. Existing depth processing/reset behavior is unchanged.

Episode filtering still uses the existing length threshold and truncation rule,
but boundaries come from episode IDs, not sampled mutable `dones`. All sensor,
label, and provenance tensors use the same slices. `--frac` selects original
environments and retains their selected episode columns together.

The compiler splits all input files jointly. Every `(source_id, original_env_id)`
is indivisible, even across episodes or exports. Valid explicit terrain/track
seed metadata joins groups sharing a realization. Without such metadata,
terrain-realization disjointness is **not verified**. At least five independent
groups are required for nonempty 60/20/20 partitions. Overlapping/duplicate
exports are rejected; renaming files does not change their capture identities.

Compiled tensors keep the existing names and carry one provenance entry per
observation. `sequence_ids` are JSON-encoded `(capture_id, original_env_id,
episode_id)` tuples. Merging orders observations chronologically per source/env
and keeps explicit sequence IDs; differing `per_eps` values become 0 instead of
borrowing the first file's length. EMA/Bayes receive integer encodings of these
IDs and reset at every episode boundary.

`split_manifest.json` records sources, original environment IDs, selected raw
columns, seed groups, and checks for lengths, chronology, contiguous sequences,
and split disjointness. The frozen paper evaluator repeats validation before
training/evaluation and stores checks in its existing `manifest.json`. Structural
and ordered test may overlap each other, but neither may overlap train/validation
source/environment groups. No existing metric, filter, training setting, plot,
or timing-accounting definition changes.

## Explicit legacy fallback

Historical files cannot establish true reset boundaries or original environment
groups after episode filtering. By default they are rejected. To knowingly use
unverified historical data, pass `--allow-legacy-provenance` to the compiler and
paper evaluator (Python APIs: `allow_legacy=True`). Each raw column becomes a
pseudo-environment/pseudo-episode, qualified by its source path; old compiled
files lacking explicit IDs use their own `per_eps`, or a single sequence if none
is available. These inferred IDs do not prove real episode boundaries or prevent
leakage already present in the old data. Manifests mark verification false;
warnings explain the fallback. Recollect/recompile for verified paper results.
