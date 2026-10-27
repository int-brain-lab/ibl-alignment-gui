(howto-compute-features)=
# How to compute the features for a probe

This guide shows you how to compute the per-channel electrophysiology features that the Channel
Prediction models run on.

## Prerequisites

- The `ephysatlas` extra installed (see {ref}`howto-install-extras`).
- The raw SpikeGLX AP and LF binary files of the probe.

## Steps

1. Run `compute_features_from_file` on the probe's raw data:

   ```python
   from pathlib import Path

   from ephysatlas.feature_computation import compute_features_from_file

   ap_file = Path('/path/to/probe00/data.ap.bin')
   lf_file = Path('/path/to/probe00/data.lf.bin')
   output_dir = Path('/path/to/probe00/features')

   compute_features_from_file(ap_file=ap_file, lf_file=lf_file, output_dir=output_dir)
   ```

   This writes a parquet file of per-channel features into `output_dir`.

2. Repeat for every probe you want to run the models on.

3. Point the GUI at the file (see {ref}`howto-give-shanks-features`).

The features must be of the vintage the model was trained on. If they are not, see
{ref}`howto-fix-vintage-mismatch`.

## Related

- {ref}`ref-input-datasets`: the features file
