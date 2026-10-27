(howto-prepare-data)=
# Prepare data

These guides show you how to convert your electrophysiology and histology data into files the GUI
can read. The full list of files the GUI understands, and where each one goes, is in
{ref}`ref-input-datasets`.

To run the GUI you need three things:

Electrophysiology data
: Spike sorting output (`spikes.*`, `clusters.*`, `channels.*`), processed electrophysiology
  (the RMS and power spectrum files, `_iblqc_*`), or both.

The position of each channel on the probe
: So that the GUI can place the electrophysiology along the probe's depth. This comes from the
  channel positions in the spike sorting output (`channels.localCoordinates`) or from the SpikeGLX
  metadata file (`.ap.meta`). See {ref}`ref-probe-geometry`.

The traced probe track
: The trajectory (3D coordinates) of the probe traced in your histology, in IBL coordinates, as
  `xyz_picks.json`.

Anything more gives you more plots and makes landmarks easier to find: raw data adds snippets of
the raw signal, and your own histology images can be shown in the Slice figure. The features used
by the channel prediction models can also be computed from the raw data alone (see
{ref}`howto-compute-features`).

```{toctree}
:maxdepth: 1

convert_kilosort
convert_spikeinterface
export_track_brainreg
export_track_lasagna
convert_ccf_coordinates
use_histology_images
open_without_spike_sorting
```

## Tracing tools

The trajectory can come from any tool that traces the probe track in histology registered to the
Allen atlas, such as brainreg-segment, Lasagna, SHARP-Track or HERBS. If your tool is not covered by
a guide above, export the traced points in Allen CCF coordinates and follow
{ref}`howto-convert-ccf-coordinates`.
