# Closed-loop diagnostics (eligible_episode_segment_v2)

Episode slots are reserved per track, including when multiple environments share a
track. Only those episodes contribute outcomes or recognition counters. Headline
recognition/wrong-skill metrics are valid-classification-frame weighted over completed
eligible episodes; success and distance are episode weighted. Incomplete eligible
episodes are retained for diagnostic inspection with `step_cap`, not counted as
completed outcomes. Inference timing retains the full simulated batch.

Replay captures oracle and learned-router trajectories, split by explicit environment
and episode IDs. Invalid depth/proprioception holds the previous output without
updating EMA/Bayes. Auto-reset observations are never attached to the preceding
episode. The terminal position/outcome is captured separately. Physical boundary
crossings use pre-reset control-step positions and interpolated crossing times.

First recognition is the first valid matching prediction within the observed target
GT segment, excluding its end. Persistent switching requires two consecutive valid
classification ticks; an anticipatory run must survive into the target segment.
Early runs that revert before the boundary are recorded separately and cannot hide
a miss. The diagnostic window is +/-5 classification ticks. An unfinished segment
without a match is censored, not a confirmed miss. Missing offsets/rates are null.
Legacy headline matched-delay fields remain available; corrected recognition and
signed persistent-switch offsets live in `transition_lead_lag*.csv` with the metric
version, denominators, reversions and censoring flags.

`locomotion_transition_events.csv` records observed approaches (within 2 m), crossings,
and traversal outcomes. Traversal requires reaching the destination cell end (or
course completion on the last cell); timeout alone does not establish traversal.
Unfinished crossings, retreats and unreached boundaries are distinguished. The
legacy pair `success_rate` explicitly means episode success on a course containing
that pair, not successful traversal of that pair; use `traversal_success_rate` and its
resolved denominator for traversal analysis.

`replay_diagnostic_bundles/*.pt` contains one source episode per file: captured depth,
proprioception/pose, commands, provenance, labels, validity, boundary events and both
frozen classifiers' logits/probabilities, EMA scores and Bayesian beliefs/selections.
Load trusted local bundles using `torch.load(path, map_location="cpu", weights_only=False)`.
These tensors and metadata suffice to recreate tables, timelines and depth examples
without datasets, checkpoints, simulation or inference. Existing figures reload CSVs;
source-specific plots are under `replay_by_source/`. No timeline joins episodes.
Online-vs-replay label mismatch counts are saved for the deployed source classifier.

Legacy replay lacking episode IDs is reported unavailable, not stitched together.
Hardware acquisition/frame IDs are unavailable; buffer-age information is marked as
such where present. Offline deltas are only computed for matching canonical label
spaces. Online source-method comparisons are frame weighted and stratified by source,
difficulty, transition pair, outcome and transition/steady observations: they are not
causal evidence that a perception difference caused a locomotion failure.
