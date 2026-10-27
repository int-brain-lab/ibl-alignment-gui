(howto-convert-spikeinterface)=
# How to convert data with SpikeInterface

This guide shows you how to create the GUI's input files from any recording system or spike sorter
that [SpikeInterface](https://spikeinterface.readthedocs.io/) can read, such as Open Ephys.

## Prerequisites

- SpikeInterface installed in your analysis environment.
- A `SortingAnalyzer` for the sorting, with the `templates`, `spike_amplitudes` and
  `quality_metrics` extensions computed.

## Steps

1. Build the `SortingAnalyzer` and compute the extensions the exporter needs, following the
   SpikeInterface documentation.

2. Export it with `export_to_ibl_gui`. Pass the LF band recording as `lfp_recording` to also get
   the LF RMS and power spectrum plots:

   ```python
   from spikeinterface.exporters import export_to_ibl_gui

   export_to_ibl_gui(
       sorting_analyzer,
       output_folder='/path/to/output',
       lfp_recording=lfp_recording,
   )
   ```

   The AP band RMS is computed only if the analyzer has its recording attached.

3. Check that the output folder holds `spikes.*`, `clusters.*` and `channels.*` files, and the
   `_iblqc_*` files.

4. In the session YAML, use the output folder as both the `spike_sorting` and the
   `processed_ephys` folder (see {ref}`howto-single-probe-yaml`).

The exporter's `good_units_query` sets which units are labelled good in `clusters.metrics.csv`,
which the unit filters use (see {ref}`howto-filter-units`).

## Related

- [`export_to_ibl_gui` source](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/exporters/to_ibl.py)
- {ref}`ref-input-datasets`
