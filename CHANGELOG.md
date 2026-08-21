Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-21

Adds region prediction from electrophysiology features, a 3D viewer, session YAML configuration and
an Allen/Code Ocean workflow, and makes the IBL online dependencies optional so the GUI can be used
entirely offline.

### Added
- Channel Prediction plugin, predicting the brain region of each channel from its electrophysiology
  features with either an inference classifier or a spatial encoder (`ephysatlas` extra).
- 3D viewer of the probe trajectory and clusters, built on datoviz.
- Session YAML configuration, for sessions whose datasets are spread across several folders, or
  that hold several probes or two configurations.
- Allen/Code Ocean workflow, with an `alignment-gui-allen` entry point and alignments read from and
  written to the Allen DocDB (`allen` extra).
- Save Progress (`Shift+S`), so an alignment that has not been uploaded can be recovered.
- Region Plots menu (`Alt+5`) offering the Allen, Beryl and Cosmos mappings.
- Range Controller plugin for setting the range of each plot type.
- Raw LF plots, tip scatter and scale factor plots, and applying the QC of one shank to all of them.
- Loading an insertion directly by id, with `alignment-gui-ibl -p <pid>`.
- Documentation for the Channel Prediction plugin, the datasets and the keyboard shortcuts.

### Changed
- `ibllib` is now optional, installed with the `ibl` extra; offline mode runs without it.
- The package was reorganised: imports of `app.app_controller`, `app.app_view`, `utils.utils` and
  `plugins.features_3d` must be updated.
- A Qt binding, `pandas`, `scipy`, `pyyaml` and `requests` are now declared explicitly.
- The stored (resolved) alignment is loaded by default, and the unit filter shows all units.
- Data loading and plot building run on a background thread, with a progress dialog.

### Fixed
- Alignments could differ between the two configurations of a shank after an upload, and previous
  alignments saved locally were not loaded.
- Probes read from a session YAML were ordered non-deterministically, so the shank tabs could
  change between runs.
- Sessions with no histology path, with an explicit `processed_ephys` but no `raw_ephys`, or with an
  empty or malformed YAML, failed to load.
- The histology download could hang indefinitely, and the TIFF reader failed to read stacks.

### Removed
- The offset button, `CollectionData` (replaced by the paths resolved from the session YAML), and
  the unused `backend` YAML field.

## [0.1.0] - 2025-02-03
### Added
- Initial release of the IBL Alignment GUI.
