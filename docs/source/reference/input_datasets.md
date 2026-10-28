(ref-input-datasets)=
# Input datasets

This page lists the files the GUI reads, grouped by the folder each group is read from. The folders
are given in the session YAML under the key shown for each group (see {ref}`ref-session-yaml`). When
a single data folder is opened instead of a YAML file, every dataset is read from that one folder.

To produce these files from your own recordings, see {ref}`howto-prepare-data`. The files the GUI
writes are described in {ref}`ref-output-files`.

| Datasets | YAML key | Contents |
|---|---|---|
| `spikes.*`, `clusters.*`, `channels.*` | `spike_sorting` | Spike sorting output |
| `_iblqc_ephysTimeRms*`, `_iblqc_ephysSpectralDensityLF.*` | `processed_ephys` | RMS and power spectra computed from the raw data |
| `*.ap.meta`, `*.ap.bin`, `*.lf.bin` | `raw_ephys` | Raw SpikeGLX recordings and their metadata |
| Histology volumes | `histology` | Histology images registered to the Allen CCF |
| `*xyz_picks*.json` | `picks` | Traced probe trajectory |
| Per-channel features (`.pqt`) | `features` | Electrophysiology features for the region prediction models |

## Spike sorting

Read from the `spike_sorting` folder.

`spikes.amps`
: **Units** V · **Type** double · **Dimensions** [number of spikes] · **Format** npy

  Amplitude of each spike.

`spikes.clusters`
: **Units** index · **Type** int · **Dimensions** [number of spikes] · **Format** npy

  Cluster assignment of each spike, counting from 0. Can be used to index the `clusters.*`
  attributes directly.

`spikes.depths`
: **Units** µm · **Type** double · **Dimensions** [number of spikes] · **Format** npy

  Depth along the probe of each spike, computed from the waveform centre of mass. 0 is the probe
  tip, positive upwards.

`spikes.times`
: **Units** s · **Type** float64 · **Dimensions** [number of spikes] · **Format** npy

  Time of each spike, relative to the start of the experiment.

`clusters.channels`
: **Units** index · **Type** int · **Dimensions** [number of clusters] · **Format** npy

  The channel each cluster is assigned to, used to place the clusters along the probe.

`clusters.metrics`
: **Format** csv or pqt · **Dimensions** [number of clusters]

  Cluster-level quality metrics. No field is required, but two of them enable the unit filters:
  `ks2_label` enables the `KS good` and `KS mua` filters, and `label` enables the `IBL good`
  filter. If the field a filter needs is missing, the GUI logs a warning and shows all units.

`clusters.peakToTrough`
: **Units** ms · **Type** double · **Dimensions** [number of clusters] · **Format** npy

  Duration of the template waveform, from peak to trough (`peak_sample - trough_sample`).
  Negative if the peak comes first.

`clusters.waveforms`
: **Units** V · **Type** single · **Dimensions** [number of clusters, number of time samples,
  number of selected channels] · **Format** npy

  Template waveforms, stored only for the channels closest to the peak channel.

`channels.localCoordinates`
: **Units** µm · **Type** float · **Dimensions** [number of channels, 2] · **Format** npy

  Position of each channel in the probe coordinate system. The first column is across the width
  of the probe, the second is depth, with 0 at the deepest site and positive above it. Used for
  the probe geometry (see {ref}`ref-probe-geometry`).

`channels.rawInd`
: **Units** index · **Type** int · **Dimensions** [number of channels] · **Format** npy

  Index of each channel in the raw recording file, counting from 0. Can be used to index the raw
  AP and LF binary files directly.

## Processed electrophysiology

Read from the `processed_ephys` folder. If no `processed_ephys` path is given, it is read from the
`raw_ephys` folder.

`_iblqc_ephysSpectralDensityLF.power`
: **Units** V²/Hz · **Type** float32 · **Dimensions** [number of frequencies, number of channels]
  · **Format** npy

  Power spectral density of the LF band of every channel.

`_iblqc_ephysSpectralDensityLF.freqs`
: **Units** Hz · **Type** float32 · **Dimensions** [number of frequencies] · **Format** npy

  Frequencies of the power spectral density.

`_iblqc_ephysTimeRmsAP.rms`
: **Units** V · **Type** float32 · **Dimensions** [number of time windows, number of channels] ·
  **Format** npy

  RMS amplitude of the AP band over time.

`_iblqc_ephysTimeRmsAP.timestamps`
: **Units** s · **Type** float32 · **Dimensions** [number of time windows] · **Format** npy

  Times of the AP RMS windows, relative to the raw binary file.

`_iblqc_ephysTimeRmsLF.rms`
: **Units** V · **Type** float32 · **Dimensions** [number of time windows, number of channels] ·
  **Format** npy

  RMS amplitude of the LF band over time.

`_iblqc_ephysTimeRmsLF.timestamps`
: **Units** s · **Type** float32 · **Dimensions** [number of time windows] · **Format** npy

  Times of the LF RMS windows, relative to the raw binary file.

## Raw electrophysiology

Read from the `raw_ephys` folder. These files are optional, but without them the raw data plots are
not available.

`*.ap.meta`
: **Format** SpikeGLX metadata

  Metadata of the AP band. Used for the probe geometry, and the only source of it when there is no
  spike sorting (see {ref}`ref-probe-geometry`).

`*.ap.bin`, `*.lf.bin`
: **Format** SpikeGLX binary

  Raw AP and LF recordings. Short snippets are read to display the raw data plots.

## Histology

Read from the `histology` folder. Optional: without histology images, the alignment is done
against the Allen atlas template and annotation, which are always available.

The folder is searched for NRRD files first, then TIFF files.

NRRD
: **Dimensions** [AP, DV, ML] voxels · registered to the 25 µm Allen CCF

  A file with `RD` in its name is loaded as the red channel, and one with `GR` in its name as the
  green channel.

TIFF (`.tif` or `.tiff`)
: **Dimensions** [ML, DV, AP] voxels · registered to the 25 µm Allen CCF

  A file with `RD` or `C1` in its name is loaded as the red channel, and one with `GR` or `C0` in
  its name as the green channel. `C0` and `C1` can be used when the channels are not named by
  colour.

Each histology volume found adds an entry to the `Slice Plots` menu. Only one red and one green
volume are loaded. The colour sets the volume's name in the menu.

## Probe trajectory

Read from the `picks` folder.

`xyz_picks`
: **Units** µm · **Type** double · **Dimensions** [number of points, 3] · **Format** json

  Points along the traced probe track, as ML, AP, DV coordinates relative to bregma (see
  {ref}`explanation-coordinate-systems`). The file is matched as `*xyz_picks.json`, so a prefix
  may be added. For a probe with more than one shank there is one file per shank, matched as
  `*xyz_picks_shank<N>.json`, counting from 1.

  ```json
  {"xyz_picks": [[-2863, -3999, -643], [-2813, -3975, -842], [-2788, -4025, -1193]]}
  ```

## Electrophysiology features

The path to a single file, given by the `features` key. Optional, and only used by the Channel
Prediction plugin (see {ref}`ref-channel-prediction`).

`features`
: **Format** parquet · **Rows** one per channel

  Per-channel electrophysiology features computed with [`ibleatools`](https://github.com/int-brain-lab/ibleatools). A `channel` column holds the
  raw-data channel index. For a probe with more than one shank, one file can hold every shank's
  channels: the rows are split per shank by matching `channel` against `channels.rawInd`.
