(howto-compute-features)=
# How to compute the features for a probe

This guide shows you how to compute the per-channel electrophysiology features that the Channel
Prediction models run on.

## Prerequisites

- The `ephysatlas` extra installed (see {ref}`howto-install-extras`).
- The raw SpikeGLX AP and LF files of the probe (`.bin` or `.cbin`), each with its `.meta` file.

## Steps

1. Compute the features with [`ibleatools`](https://github.com/int-brain-lab/ibleatools), following
   its guide
   [How to compute features from SpikeGLX AP/LF files](https://int-brain-lab.github.io/ibleatools/how-to/compute-features-spikeglx.html).
   It gives a table of features with one row per channel.

2. Save the table as a parquet file, for example `df.to_parquet('/path/to/probe00/features.pqt')`.
   Parquet keeps the table's attributes, including the features' vintage (`features_version`),
   which the GUI checks against the model.

3. Repeat for every probe you want to run the models on.

4. Point the GUI at the file (see {ref}`howto-give-shanks-features`).

:::{note}
The published models need all four feature families. The `waveforms` family needs the full
install of `ibleatools` (`ibleatools[full]`). The feature computation also assumes a single-shank
probe: on a multi-shank recording some features can fail or mix the shanks.
:::

The features must be of the vintage the model was trained on. If they are not, see
{ref}`howto-fix-vintage-mismatch`.

## Related

- {ref}`ref-input-datasets`: the features file
