Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a2] - 2026-10-30

### Added
- After each fit, the depths of the reference line pairs along the probe and along the track, and
  the offset between them, are printed in the terminal
  ([#17](https://github.com/int-brain-lab/ibl-alignment-gui/issues/17)).
- Channel prediction models in a new `Prediction Plots` menu (`Alt+5`), shown in a panel beside
  the ephys plots: two region decoders (XGBoost and ProbeTransformer) and an alignment model that
  aligns the recording to the histology track, with a result window and a confidence view. They
  are loaded from the Hugging Face Hub, and `Shift+P` shows all predictions side by side.
- Documentation reorganised into tutorials, how-to guides, explanation and reference.

### Changed
- The `Region Plots` menu is renamed `Set Mapping` (`Alt+M`), and the chosen mapping applies to all
  histology panels.
- The ephys-atlas features come with the session: the `features` dataset in the session YAML, a
  features file in the data folder, or Alyx online. The models run on the features of their own
  vintage. Without the `ephysatlas` extra the feature and prediction menus are hidden, and without
  features they are greyed out.
- Session YAML: `output` must always be given, and `raw_ephys` no longer falls back to
  `spike_sorting`.

### Fixed
- The RMS and `Firing Rate` image plots are drawn from the recording's first timestamp rather
  than from 0 s, so they no longer appear blank when the recording does not start at 0 s
  ([#15](https://github.com/int-brain-lab/ibl-alignment-gui/issues/15)).
- The configuration dropdown is offered for every session with two configurations, not only
  when the GUI is started with one: a two-configuration YAML opened from `...` now shows it,
  and a single-configuration session hides it.
- A session YAML with two `configurations` failed to load (`KeyError` on its first probe); both
  configurations and `both` are now offered in the configuration dropdown.
- A session without spike sorting now opens, with the probe geometry read from the `.ap.meta` file
  in `raw_ephys`, which such a session must give; it used to fail on loading. The `Filter Plots`
  menu is greyed out without spike sorting.
- Saving progress or uploading no longer fails when the `output` folder does not exist yet; it is
  created.

### Removed
- `Plugins -> Channel Prediction -> Load features file…`. The features come with the session: the
  YAML's `features` dataset, a features file in the data folder, or the Alyx table online.

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
