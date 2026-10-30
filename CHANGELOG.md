Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Opening a data folder picks up the per-channel features file in it: the one parquet file with
  `features` in its name.
- Documentation reorganised into tutorials, how-to guides, explanation and reference, following
  [Diátaxis](https://diataxis.fr/), and written in Markdown (MyST). New pages include tutorials on
  the sample data, guides for data preparation, session YAMLs, channel prediction and writing
  plugins, and reference pages for the menus, command line, probe geometry and output files.
  The documentation build now fails on warnings.
- Auto-alignment in the Channel Prediction plugin: the recorded channel features are warped along
  the shank's histology track onto the Ephys Atlas channel-level model's predictions
  (`ephysatlas.alignment` in ibleatools). The released model is loaded with
  `ephysatlas.load_pretrained` from the Hugging Face Hub or a local release
  (`Load alignment model (Hugging Face)…`). Two Prediction Plots entries show the result:
  `Auto-alignment` (the atlas region of each channel where the alignment placed it on the track)
  and `Auto-alignment (confidence)` (green high, red low), both along the probe. Each shank is
  aligned once for both, so switching between them does not re-align.
- The auto-alignment result window shows the histology track, the planned trajectory and the
  inferred alignment on two slices, the cost matrix with the alignment path, and the recorded vs
  predicted features along the track. It opens once when a shank is aligned;
  `Plugins -> Channel Prediction -> Show auto-alignment results` (greyed out until then) shows the
  windows again. They close when a new model, features file or session is loaded.
- `ShankView.plot_histology(..., boundaries=False)` draws a column without region boundary lines.
- A Prediction Plots menu (`Alt+5` / `Shift+Alt+5`) shows a channel prediction on its own panel
  in the Ephys figure, after the probe plot (or the feature plot in the feature view). It carries
  the reference lines, and labels its predicted regions with their acronyms. `None` removes the
  panel.
- `All predictions (side by side)` in Prediction Plots (`Shift+P`) replaces the image, line and
  probe plots with one panel per prediction option of the loaded models, computed in one
  background task; the panels of a model that is not loaded say so.

### Changed
- Session YAML: `output` must always be given, so the results are never written to a folder that
  was not chosen. `picks` is still read from `spike_sorting` when not given, and `processed_ephys`
  from `raw_ephys`, then `spike_sorting`. `raw_ephys` no longer falls back to `spike_sorting`.
- The `Feature Plots` and `Prediction Plots` menus and `Plugins -> Channel Prediction` are only
  shown with the `ephysatlas` extra, and are greyed out when the session has no ephys-atlas
  features. Hovering over a greyed-out menu says how to provide a features file.
- The `Auto-alignment` regions view takes each channel's region at the position the alignment
  placed it (`AlignmentResult.channel_xyz`), instead of reading the probe depth of every track
  sample off a straight line fitted through the alignment. Its boundaries now follow the
  alignment's stretching and compression; the view and the confidence view cover the probe only.
- The model predictions are shown in the Ephys figure (Prediction Plots) instead of the reference
  histology panel: they are placed by depth along the probe, like the ephys plots. The Region
  Plots menu is renamed `Set Mapping` (`Alt+M` / `Shift+Alt+M`), moves after Prediction Plots,
  and keeps the `Allen`, `Beryl` and `Cosmos` mappings, now drawn by the GUI itself rather than
  the Channel Prediction plugin. The chosen mapping applies to all three histology panels (the
  aligned regions, the scale factors and the original regions); the 3D view and the uploaded
  results stay in the Allen mapping.
- The inference model is replaced by two released region decoders, `XGBoost region prediction`
  (`int-brain-lab/ea-decoder-channel-xgboost`) and `ProbeTransformer region prediction`
  (`int-brain-lab/ea-decoder-channel-transformer`, needs torch), both loaded with
  `ephysatlas.load_pretrained` from the Hugging Face Hub (release tag `2026_W39` by default) or a
  local release directory holding its `ephysatlas_model.json`, and run with their `predict`. Each
  has its own `Load … model (Hugging Face)…` menu entry and two Prediction Plots entries: the most
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
- An unexpected error while computing a region prediction is reported in the plot's warning and
  its traceback logged, instead of escaping (which could close the GUI). A feature table that
  cannot be loaded for the model's vintage is reported as such.
- A model whose feature vintage has no features for the insertion reports it, instead of predicting
  on the GUI's features of another vintage, and suggests loading a release of the model for the
  session's vintage instead.
- A feature table download that finds nothing, or is interrupted, is retried the next time instead
  of failing until its folder is deleted by hand.
- A region decoder runs once per shank for both of its Prediction Plots entries (the most probable
  region and the cumulative probabilities), instead of once for each.
- A model's load dialog, or a model that fails to load, is handled once per plot rather than once
  per shank; cancelling the dialog stops the plot.
- Model region options are offered offline too before the model is loaded, as online: selecting one
  opens its load dialog.

### Fixed
- A session without spike sorting now opens, with the probe geometry read from the `.ap.meta` file
  in `raw_ephys`, which such a session must give; it used to fail on loading. The `Filter Plots`
  menu is greyed out without spike sorting.
- Saving progress or uploading no longer fails when the `output` folder does not exist yet; it is
  created.

### Removed
- `Plugins -> Channel Prediction -> Load features file…`. The features come with the session: the
  YAML's `features` dataset, a features file in the data folder, or the Alyx table online.
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
