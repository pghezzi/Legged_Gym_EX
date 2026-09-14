# Frozen offline paper figures

Normal `evaluate_paper_offline_experiments_1_2` runs now save `figure_data.pt`
beside the existing outputs. The bundle contains both classifiers × seeds
0/1/2, all three selectors, per-seed metrics/confusion counts, structural and
ordered logits/probabilities, EMA scores/probabilities, Bayes posterior beliefs,
emitted class indices, and shared per-frame labels/provenance. Only selected
depth images are copied, together with actual extractor reference/residual,
Sobel/occupancy masks, feature values, calibration/settings, and IMU inputs.
Selected transition windows also include 48×64 depth thumbnails and their
source/frame identifiers.
No checkpoints or original datasets are needed to plot it.

Regenerate **all** existing and new figures, without inference or evaluation:

```bash
python -m legged_gym.scripts.depth_data_pipeline.evaluate_paper_offline_experiments_1_2 \
  --plot-only paper_offline_eval/figure_data.pt --output paper_figures
```

Create a bundle from an existing completed frozen offline run **without
retraining**, leaving its results and training-cost records untouched:

```bash
python -m legged_gym.scripts.depth_data_pipeline.evaluate_paper_offline_experiments_1_2 \
  --bundle-from-existing paper_offline_eval --output paper_figure_refresh
```

The latter loads `manifest.json`, saved extractor/standardizer, and
`artifacts/<architecture>/seed_<seed>/{classifier.pt,nn_model_args.pt}`. It uses
dataset paths from the existing manifest; override relocated paths with
`--classifier-data` and `--ordered-data`. Legacy inputs still require explicit
`--allow-legacy-provenance`. Use a separate output directory. This mode checks
the existing fixed filter settings, runs deterministic inference, and creates
figures/bundle, but never calls classifier fitting or changes cost records.

New outputs (PNG and PDF): paired `confusion_<method>_<seed>.{png,pdf}` for
structural classification and each ordered selector, plus aggregate pairs;
`depth_geometry_<example-index>`; `transition_delay_vs_false_transition_rate`;
and `timeline_sample_<sample>_<architecture>_seed_<seed>_<selector>` for every
sample and combination. `transition_depth_sample_<sample>` contains a separate
thumbnail contact sheet; the same thumbnails appear below each timeline.
The existing bar/scatter figures are retained and now also exported as PDF.
All PNG/PDF files are saved normally beside the bundle, not only inside it.
Open `figure_index.html` to browse a thumbnail gallery with full PNG/PDF links.
`manifest.json` links the bundle/checksum/figure paths; `figure_manifest.json`
records plots, checks, and unavailable inputs/examples.

## Interpretation and selection rules

- Confusions use identical native class order and a shared [0,1] scale.
  Aggregate counts are summed over seeds before row normalization. Absent GT
  classes are N/A, not zero recall.
- Experiment 1 samples 100 distinct valid structural-test images per GT class
  without replacement. Seeded random round-robin sampling spreads observations
  across source/environment groups, sequences, and local temporal bins. Invalid
  images are skipped and replaced; classes with fewer than 100 valid images
  retain all available examples and report the shortfall. Annotations use the actual
  extractor's processed image, calibrated residual, Gaussian/Sobel kernels,
  thresholds and near/center regions; they are not ground-truth obstacle sizes.
- Experiment 2 samples 100 distinct focal GT transitions without replacement.
  Half the target is reserved for windows with emitted-label disagreement among
  classifiers/seeds/selectors; the remainder uses unrestricted coverage sampling.
  Sampling spreads across source/environment, sequence/time, and terrain-pair
  strata. Each window includes an adjacent second boundary when available and
  up to ten padding frames, never crossing sequence boundaries or including an
  extra GT change. Single-boundary sequences remain eligible. The same windows
  are used for all 18 classifier/seed/selector combinations. Shortfalls are logged.
  This disagreement-enriched selection is diagnostic, not an unbiased performance
  sample: reported metrics still use the complete test set.
- Sampling defaults (seed 42, 100 images/class, 100 transitions) are in
  `paper_figure_bundle.PLOT_CONFIG` and saved in the bundle. Thumbnails include
  window endpoints, boundary frames and preceding frames, plus evenly spaced
  context (up to eight). Source/frame identifiers are saved for every thumbnail.
- Instantaneous evidence, EMA softmax of smoothed logits, Bayes beliefs,
  emitted labels and GT are distinct traces. “Selected skill” is the requested
  terrain-associated skill; this offline experiment does not execute locomotion.
- New delay/FTR scatter uses segment-bounded v2 matched delay in classification
  frames. Small labeled points are seeds; error bars are sample standard
  deviations across finite paired seed measurements, not confidence intervals.
  Missing-delay combinations are reported. Legacy figures keep their original
  metrics and aggregation unchanged.

Rendering all samples can produce 1,800 timeline figures, 100 contact sheets,
and 100 annotated figures per class, each in PNG and PDF, so allow time/disk
space. Plot-only mode supports older bundles, but cannot create missing images;
use `--bundle-from-existing` above to populate the expanded sample without retraining.

The bundle is trusted local PyTorch data: only load files from trusted runs.
