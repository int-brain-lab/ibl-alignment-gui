Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
