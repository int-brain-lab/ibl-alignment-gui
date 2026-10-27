Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Documentation reorganised into tutorials, how-to guides, explanation and reference, following
  [Diátaxis](https://diataxis.fr/), and written in Markdown (MyST). New pages include tutorials on
  the sample data, guides for data preparation, session YAMLs, channel prediction and writing
  plugins, and reference pages for the menus, command line, probe geometry and output files.
  The documentation build now fails on warnings.
- Auto-alignment in the Channel Prediction plugin: the recorded channel features are warped along
  the shank's histology track onto the Ephys Atlas channel-level model's predictions
  (`ephysatlas.alignment` in ibleatools). The released model is loaded with
  `ephysatlas.load_pretrained` from the Hugging Face Hub or a local release
  (`Load alignment model (Hugging Face)…`). Two Region Plots entries show the result:
  `Auto-alignment` (the atlas regions along the track, at the aligned probe depths) and
  `Auto-alignment (confidence)` (green high, red low). Each shank is aligned once for both, so
  switching between them does not re-align.
- The auto-alignment result window shows the histology track, the planned trajectory and the
  inferred alignment on two slices, the cost matrix with the alignment path, and the recorded vs
  predicted features along the track. It opens once when a shank is aligned;
  `Plugins -> Channel Prediction -> Show auto-alignment results` (greyed out until then) shows the
  windows again. They close when a new model, features file or session is loaded.
- `ShankView.plot_histology(..., boundaries=False)` draws a column without region boundary lines.

### Changed
- The inference model is replaced by two released region decoders, `XGBoost region prediction`
  (`int-brain-lab/ea-decoder-channel-xgboost`) and `ProbeTransformer region prediction`
  (`int-brain-lab/ea-decoder-channel-transformer`, needs torch), both loaded with
  `ephysatlas.load_pretrained` from the Hugging Face Hub (release tag `2026_W39` by default) or a
  local release directory holding its `ephysatlas_model.json`, and run with their `predict`. Each
  has its own `Load … model (Hugging Face)…` menu entry and two Region Plots entries: the most
  probable region and the `(cumulative)` probabilities. The `void_fluid` class is drawn with its
  own colour.
- The region decoders and auto-alignment model run on the ephys-atlas feature table matching the
  model's vintage (its manifest `vintage`, e.g. `2026_W39`), not the latest table the GUI loads for
  its feature plots, whose feature set can differ. Auto-alignment takes the feature list from the
  loaded release rather than the code's `FEATURE_LIST`, so older release tags work. Missing feature
  columns are reported in a dialog instead of raising, with the same message for every model.
- Auto-alignment reads its features like the region decoders, from the GUI's ephys-atlas feature
  tables (or the features file), instead of downloading the feature release separately into
  `~/ephys-atlas/data`, which can be deleted.
- A model whose release names no feature vintage (`vintage` in its `ephysatlas_model.json`) is
  refused, rather than run on the GUI's features (region decoders) or on the table named after its
  release tag (auto-alignment).
- A features file chosen with `Load features file…` is used by both models, also for an IBL
  insertion (auto-alignment previously used the model's feature release instead).
- An unexpected error while computing a region prediction is reported in the plot's warning and
  its traceback logged, instead of escaping (which could close the GUI). A feature table that
  cannot be loaded for the model's vintage is reported as such.
- A model whose feature vintage has no features for the insertion reports it, instead of predicting
  on the GUI's features of another vintage, and suggests loading a release of the model for the
  session's vintage instead.
- A feature table download that finds nothing, or is interrupted, is retried the next time instead
  of failing until its folder is deleted by hand.
- A region decoder runs once per shank for both of its Region Plots entries (the most probable
  region and the cumulative probabilities), instead of once for each.
- A model's load dialog, or a model that fails to load, is handled once per plot rather than once
  per shank; cancelling the dialog stops the plot.
- Model region options are offered offline too before the model is loaded, as online: selecting one
  opens its load dialog.

### Removed
- The `Inference Model` / `Inference Cumulative` region entries and `Load inference model…`, the
  S3 inference models and local models in the old `folds/FOLD0X/meta.yaml` layout (these no longer
  load with the current ibleatools).

## [0.1.0a1] - 2026-08-21

### Added
- Channel prediction plugin, to predict channel location using inference or spatial encoding models (available by installing with `ephysatlas` extra).
- Allen workflow for performing alignments in imaging space and reading and writing alignments to the Allen DocDB (available by installing with `allen` extra).
- Save Progress (`Shift+S`), so an alignment that has not been uploaded can be recovered.
- Region Plots menu (`Alt+5`) offering different region mappings.
- Plots - raw LF plots, feature plots (with `ephysatlas` extra) and region plots (with `ephysatlas` extra).
- Loading an insertion directly by id, with `alignment-gui-ibl -p <pid>`.
- Documentation for the Channel Prediction plugin, the datasets and the keyboard shortcuts.

### Changed
- The package was reorganised: imports of `app.app_controller`, `app.app_view`, `utils.utils` and
  `plugins.features_3d` must be updated.
- `ibllib` is now optional, installed with the `ibl` extra; offline mode runs without it.
- 3D viewer now uses `ibl-datoviz` as the backend
- Data loading and plot building run on a background thread, with a progress dialog.

### Fixed
- Probes read from were randomly ordered, now they are sorted alphabetically by name.
- Sessions with no histology path in the YAML failed to load. Interpret as a session with no additional histology volumes.

### Removed
- The offset button has been removed, the same functionality can be achieved using a fit with a single reference line.

## [0.1.0a0] - 2025-02-03
### Added
- Initial release of the IBL Alignment GUI.
