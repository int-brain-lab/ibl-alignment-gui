Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- `extract_ephys` computes the RMS and power spectra on samples of the recording rather than on
  all of it: 1 s every 10 s for the AP band and 3 s every 10 s for the LF band, so it reads a
  fraction of the files, and processes the samples in parallel threads. `extract_rmsmap(...,
  sample_spacing_secs=None)` reads the whole recording as before.
- `extract_ephys` destripes the AP band before computing its RMS (300 Hz high-pass, ADC sample
  shift correction and the median of each shank subtracted).
- `extract_ephys` averages the power spectral density over the windows rather than summing it,
  so it no longer depends on the length of the recording.

### Added
- `extract_ephys` computes the LF RMS and power spectral density of NP2 recordings without an LF
  file from the AP file, decimated to 2.5 kHz.

### Fixed
- `extract_ephys` writes the RMS and power spectral density with the channels in the order they
  are stored in the raw file, and saves that order as `rawInd` with each `_iblqc_` object. Files
  made before had the channels sorted by position on the probe, so on NP2 probes the plots showed
  the wrong channels; recompute them with `extract_ephys`. Passing an open `spikeglx.Reader` to
  `rmsmap` no longer fails.
- The AP RMS of `extract_ephys` included the LFP for NP2 probes, which record the AP band
  wideband; the AP band is now high-passed at 300 Hz as part of the destriping.
- The RMS and power spectra are read in the channel order given by their `rawInd`. Local files
  without it may be in the wrong order when the raw file does not store the channels in shank
  and depth order: they are then logged, and a dialog recommends running `extract_ephys` again to
  verify them.
- The raw data snippets and the dead, noisy and outside channel plots show only the channels of
  each shank, at their depth, rather than every channel of the probe. Cached snippets from older
  versions are recomputed.
- A session YAML with a single multi-shank probe failed to load (`KeyError` on `shank_1`).
- The raw data snippets of recordings shorter than 500 s failed to load, as the last snippet
  started at the end of the file.
- The image, scatter and probe plots are shown for a session without xyz picks; before, only the
  line plots were drawn.
- Spike sorting whose `spikes` object lacks `times`, `depths` or `amps` (e.g. Kilosort output not
  converted with `extract_ephys`) is treated as missing and logged, rather than the load failing
  with a `RecursionError` in the unit filter.
- Colouring the channels on the histology slice by a probe plot failed when the spike sorting had
  fewer channels than the recording (`Number of brushes does not match number of points`), and
  could give channels the colour of another when they were stored out of depth order. The
  values are now matched to the channels by raw channel, and channels without one are grey.

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
