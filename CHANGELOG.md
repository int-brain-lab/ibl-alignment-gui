Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A SpikeInterface `SortingAnalyzer` can be opened directly as the spike sorting, detected from
  the path or set with `format: spikeinterface`. Needs the `spikeinterface` extra.
- `extract_si_qc` writes the RMS, power spectra and raw data snippets from SpikeInterface
  recordings, with the probe contact ids used to match the analyzer's channels.
- The raw data snippets are read from `processed_ephys` when `raw_ephys` holds none.

### Changed
- `extract_ephys` no longer saves the sync channel, and only `extract_data` needs the `ibl`
  extra.
- Bad channel detection keeps the two channels of a probe row in file order, which can change
  the flags of NP2 probes slightly.
- datoviz is only loaded when the 3D features view is opened, not when the GUI starts.

### Fixed
- Previous and Next show the reference lines that the fit at that step used, instead of keeping
  the lines of the latest fit. Lines added or moved since the last fit are discarded.
- Choosing another alignment from the dropdown keeps the current fit in the alignment history, so
  it can be recovered with undo, instead of overwriting it.
- A shank with no spikes, e.g. a dead shank, loads without its spike plots instead of failing the
  whole session.
- Uploading to Alyx re-reads the stored alignments first, so an alignment uploaded by someone else
  since the GUI loaded them is kept, and only replaces earlier alignments of the same user
  (`mf` no longer matches `mfaulkner`).
- The original alignment of a probe with channels above 4 mm starts from a range sized to the
  probe (up to 1.5 times the deepest channel) instead of +-6 mm, so reference lines placed above
  6 mm are no longer dropped from the fit. Probes with channels up to 4 mm are unchanged.
- The firing rate, correlation, passive and receptive field images are drawn at the depths their
  spikes were binned at. Spikes below the lowest site shifted or compressed these images before.
  Spikes localised below the probe tip (depth < 0) are left out of them.
- Replotting the probe plot replaces its colour bar instead of adding another one, and an empty
  probe plot no longer keeps the previous colour bar. The same applies to the scale factor colour
  bar on each fit and to the shared colour bars shown with both configurations.
- Opening the Range Controller or Ephys Features while it is already open brings it to the front
  instead of connecting its controls again, which made each slider change replot more than once
  and Next/Previous skip pages.
- The Range Controller closes when a new session is loaded, instead of keeping the previous
  session's shanks and failing on the next plot change.
- Closing a plugin window (Range Controller, Ephys Features, 3D features or a cluster popup)
  deletes it instead of keeping it hidden, and closing the 3D view releases its datoviz server.
  Clicking a cluster whose popup is already open brings that popup to the front.
- Dragging a range slider (Range Controller, Ephys Features) redraws the plots at most every
  100 ms and once more on release, instead of on every mouse movement.
- In Ephys Features, changing page or region selects the probe in the same column of the new
  page, so the colour slider and reset act on a probe that is shown.
- Loading a session with fewer shanks than the previous one no longer leaves empty rows and
  their splitter handles in the shank grid.
- The shank markers in the 3D view are replaced on each replot instead of being added again,
  keep a fixed colour per shank, no longer overwrite each other when shank names end in the same
  character, and are shown with both configurations whatever the configuration names.
- Reference lines hidden with Shift+H stay hidden through fits, resets and changes of mapping or
  configuration. Double-clicking to add a line shows them again, and a new probe starts with them
  shown.

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
