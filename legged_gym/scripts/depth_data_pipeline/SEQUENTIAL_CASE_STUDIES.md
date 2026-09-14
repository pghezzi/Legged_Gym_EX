# Saved Sequential Case Studies

Run the plotting entry point directly to avoid simulator initialization:

```bash
python legged_gym/scripts/depth_data_pipeline/plot_sequential_comparison_examples.py \
  /path/to/offline --case-studies --ordered-test /path/to/compiled/test.pt
```

This reads saved full-sequence predictions, `experiment_2_transitions_v2.csv`
(when available), and provenance-aligned compiled depth observations. It never
loads models, reruns inference, or resets filters. The original ten illustrative
figures remain unchanged. Case C requires their `sequential_comparison_01.json`
to verify the exact source sequence/seed.

New outputs are in `figures/depth_and_trajectories/sequential_case_studies/`:

- `main_stability_and_short_segment.pdf/png`: Cases A and B.
- `appendix_sustained_misclassification.pdf/png`: Case C.
- `case_study_figure_data.pt`: exact plotted traces, source/frame provenance,
  normalized depth observations, masks, selection/metric records, and settings.
- `candidate_metrics.json`, `chosen_metrics.json`, `selection_metadata.json`:
  deterministic ranking, full-segment matching, checks, and unavailable evidence.

Regenerate without accessing datasets, checkpoints, or the original figure bundle:

```bash
python legged_gym/scripts/depth_data_pipeline/plot_sequential_comparison_examples.py \
  unused --plot-case-bundle /path/to/case_study_figure_data.pt
```

The PDFs have a native width of 7.16 inches, embedded Times New Roman, no
Type 3 fonts, and at least 12-point text at that width. Use full page-width
inclusion without further shrinking. PNGs use 300 dpi. These are illustrative
post-hoc examples, not aggregate-performance estimates.

Metric semantics are unchanged: transition windows are +/-5 classification
frames, clipped at actual sequence boundaries; steady frames are their complement.
First match is searched through the full target segment, excluding its end.
A miss means no match anywhere in that segment. First match does not imply a
sustained or anticipatory switch. Erroneous-switch rate uses the existing false
transition definition (selection changes while GT remains unchanged, divided by
adjacent within-sequence frame opportunities). Full-sequence masks are sliced,
not recomputed at cropped edges. Appendix window error is the fraction of
plotted frames where Raw Bayes disagrees with GT, not a whole-segment miss.

Timeline colors use Stairs, Gap, Pit, and Rough (the display name for the
evaluator's `random_uniform` class); no class merging or relabeling is performed.
Gray shading indicates actual transition-window frames. Line styles distinguish
instantaneous probabilities, EMA probabilities, and Bayes beliefs. Temporal
segment length is not a physical obstacle length or evidence of control effects.
