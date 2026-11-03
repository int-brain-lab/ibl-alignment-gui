(ref-probe-geometry)=
# Probe geometry

This page explains how the GUI finds the layout of the probe: the position of each site and which
shank it is on.

## Sources

The geometry is read from two sources, when present:

Spike sorting channels
: `channels.localCoordinates` and `channels.rawInd` in the `spike_sorting` folder. Shanks are told
  apart by a gap of more than 100 µm between the x coordinates of neighbouring sites.

SpikeGLX metadata
: The `*.ap.meta` file in the `raw_ephys` folder, read with `ibl-neuropixel`. Shanks are taken
  from the shank number of each electrode in the metadata.

At least one of the two is needed; loading fails if neither is found.

When both are found, they are checked against each other: each shank must have the same number of
columns of sites, and every channel must lie within the depth range of the metadata for its
shank. If they disagree, a warning is logged and the spike sorting channels are used for
everything, since the spike sorting data is indexed by them.

## Channel order

Channels are numbered by their position in the raw recording file, counting from 0, which is not
always their order along the probe. On Neuropixels 2.0 probes the file can hold the channels in
blocks out of depth order, or interleave the shanks.

Spike sorting can list its channels in another order: data read with `ibl-neuropixel` is sorted by
shank, row and column, which on Neuropixels 2.0 swaps the two channels of each row.
`channels.rawInd` gives the raw file channel of each spike sorting channel, and the `rawInd`
attribute of the RMS and power spectrum files does the same for their columns, so the data of each
channel is found whatever order it is stored in.

## Which one is used

| Used for | Geometry |
|---|---|
| The plots in the Ephys figure | The metadata, or the spike sorting channels if there is no metadata |
| The channel locations saved on upload | The spike sorting channels, or the metadata if there are no channels |

With only the metadata, a session can be opened before it has been spike sorted. The plots that
need spike sorting are then empty.

## Supported probes

Any probe whose geometry can be read from one of the sources above. The GUI is tested with:

| Probe | Shanks |
|---|---|
| Neuropixels 1.0 (3A) | 1 |
| Neuropixels 1.0 (3B) | 1 |
| Neuropixels 2.0 single shank (2.1) | 1 |
| Neuropixels 2.0 four shank (2.4) | 4 |

For probes with more than one shank, each shank is aligned separately and needs its own traced
trajectory (see {ref}`ref-input-datasets`).
