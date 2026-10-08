Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Auto-alignment in the Channel Prediction plugin (`Region Plots -> Auto-alignment`): the recorded
  channel features are warped along the shank's histology track onto the Ephys Atlas channel-level
  model's predictions (`ephysatlas.alignment` in ibleatools). The released model is loaded with
  `ephysatlas.load_pretrained` from the Hugging Face Hub or a local release
  (`Load alignment model (Hugging Face)…`). A result window shows the histology track, the planned
  trajectory and the inferred alignment on two slices, the cost matrix with the alignment path, and
  the recorded vs predicted features along the track. Selecting the option again re-opens a closed
  result window without re-aligning.
- `Plugins -> Channel Prediction -> Auto-alignment colours`: colour the auto-alignment column by the
  predicted regions or by the alignment confidence (green high, red low).
- `ShankView.plot_histology(..., boundaries=False)` draws a column without region boundary lines;
  `AppView.triggering_menu_option` tells callbacks a programmatic menu selection from a user's.

### Changed
- The inference model (`Load inference model (Hugging Face)…`) is the released region classifier
  `int-brain-lab/ea-decoder-channel-xgboost`, loaded with `ephysatlas.regionclassifier.RegionClassifier`
  from the Hugging Face Hub (repository and release tag, by default the `2026_W26` vintage tag) or a
  local release directory holding its `ephysatlas_model.json`. Its `void_fluid` class is now drawn with its own colour.
- The inference and auto-alignment models run on the ephys-atlas feature table matching the
  model's vintage (its manifest `vintage`, e.g. `2026_W26`), not the latest table the GUI loads for
  its feature plots, whose feature set can differ. Auto-alignment takes the feature list from the
  loaded release rather than the code's `FEATURE_LIST`, so older release tags work. Missing feature
  columns are reported in a dialog instead of raising.
- A features file chosen with `Load features file…` is used by both models, also for an IBL
  insertion (auto-alignment previously used the model's feature release instead).

### Removed
- The S3 inference models and local models in the old `folds/FOLD0X/meta.yaml` layout; these no
  longer load with the current ibleatools.

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
