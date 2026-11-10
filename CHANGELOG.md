Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- The docs give the version tags of the channel- and unit-level models (`0.0.1`, ...) as examples;
  the alignment model's default release is ibleatools' `ephysatlas.alignment.CHANNEL_MODEL_REVISION`,
  `0.0.1` since ibleatools 0.13.0.

## [0.1.0a5] - 2026-11-09

### Added
- A SpikeInterface `SortingAnalyzer` can be opened directly as the spike sorting (`spikeinterface`
  extra).
- `extract_si_qc` writes the RMS, power spectra and raw data snippets from SpikeInterface
  recordings.
- The raw data snippets are read from `processed_ephys` when `raw_ephys` holds none.

### Changed
- `extract_ephys` no longer saves the sync channel, and only `extract_data` needs the `ibl` extra.
- Bad channel detection keeps the channels of a probe row in file order, which can slightly change
  the flags of NP2 probes.
- The original alignment is sized to the probe, up to 1.5 times the deepest channel, instead of
  +-6 mm, so reference lines above 6 mm are kept. Probes with channels up to 4 mm are unchanged.
- The spike-based images are drawn at the depths their spikes were binned at, leaving out spikes
  below the probe tip.
- Previous and Next show the reference lines of the fit at each step.
- Sessions with more than four shanks stop with an error saying so.
- datoviz is only loaded when the 3D view is opened.

### Fixed
- Uploading to Alyx keeps alignments uploaded by others since the GUI loaded them, and only
  replaces earlier alignments of the same user.
- Sessions load when a shank has no spikes or no picks, a folder has only an `ap.meta`, a yaml
  probe has no `datasets:`, or a SortingAnalyzer's processed ephys is from another recording
  (shown without its RMS and snippets).
- Choosing another alignment keeps the current fit in the undo history.
- Colour bars, plugin windows, their signal connections and the 3D markers no longer build up over
  a session.
- Hidden reference lines stay hidden, range sliders redraw at most every 100 ms, Ephys Features
  keeps its selection on the page shown, and loading fewer shanks leaves no empty grid rows.

## [0.1.0a4] - 2026-11-03

### Added
- `extract_ephys` computes the LF RMS and power spectrum of NP2 recordings without an LF file
  from the AP file.

### Changed
- `extract_ephys` computes the RMS and power spectra on samples of the recording, in parallel,
  destripes the AP band before its RMS (which removes the LFP from the AP band of NP2 probes),
  and averages the power spectrum over windows rather than summing it.

### Fixed
- `extract_ephys` saved the RMS and power spectra with the channels sorted by position on the
  probe rather than in file order, so NP2 plots showed the wrong channels. It now saves them in
  file order with a `rawInd`; recompute older files. Local files without `rawInd` that may be
  affected are reported when the session loads.
- The raw data snippets and channel plots show each shank's own channels, at their depth.
- The image, scatter and probe plots are shown for sessions without xyz picks.
- Colouring the slice channels by a probe plot failed, or used the wrong channels, when the spike
  sorting had fewer channels than the recording or listed them in another order.
- Loading failed for a session YAML with a single multi-shank probe, for recordings shorter than
  500 s, and for spike sorting without `spikes.times`, `depths` or `amps`.

## [0.1.0a3] - 2026-11-02

### Changed
- The alignment model loads its own release (`2026_W45`) by default instead of the feature vintage
  it was trained on; the vintage now comes only from the release's `ephysatlas_model.json`.
- A local features file of another vintage than the model's is refused, as Alyx features already
  were. The error names both vintages and how to resolve the mismatch.
- The prediction panels have short titles, such as `XGBoost prediction`, in the GUI's usual font.

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
