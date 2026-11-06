(howto-convert-spikeinterface)=
# How to load a SpikeInterface SortingAnalyzer

This guide shows you how to open a SpikeInterface
[`SortingAnalyzer`](https://spikeinterface.readthedocs.io/) in the GUI, for any recording system
or spike sorter that SpikeInterface can read, such as SpikeGLX or Open Ephys. The GUI reads the
analyzer directly, so the spike sorting does not need converting. The RMS, power spectrum and raw
data plots come from files you write once with the `extract_si_qc` script.

## Prerequisites

- The GUI installed with the `spikeinterface` extra (see {ref}`ref-optional-dependencies`).
- A saved `SortingAnalyzer`, as a binary folder or a `.zarr`, with a single segment.
- The raw recording the analyzer was sorted from, readable by a SpikeInterface reader.

## Steps

1. Compute the analyzer extensions the GUI reads. `templates` and `spike_amplitudes` are needed
   for the spikes and clusters; the others are used when they are there:

   ```python
   analyzer.compute(['random_spikes', 'templates', 'spike_amplitudes'])
   # Optional: spike depths, waveform durations and the IBL good label
   analyzer.compute(['spike_locations', 'template_metrics', 'quality_metrics'])
   ```

   | Extension | Used for | Without it |
   |---|---|---|
   | `spike_locations` | The depth of each spike | The depth of the unit's main channel |
   | `template_metrics` | The peak-to-trough duration of each unit | Measured on the template at its main channel |
   | `quality_metrics` | The `IBL good` unit filter | The filter shows all units |

   A unit is `IBL good` when `amplitude_median < -40 and isi_violations_ratio < 0.5 and
   amplitude_cutoff < 0.2`, as in SpikeInterface's `export_to_ibl_gui`. The `KS good` and
   `KS mua` filters use the sorting's `KSLabel` property, which Kilosort sortings read with
   SpikeInterface have (see {ref}`howto-filter-units`).

2. Write the RMS, power spectra and raw data snippets with `extract_si_qc`. Open the raw
   recording with the SpikeInterface reader of your acquisition system, as read from disk, with
   no filtering or referencing, and choose how the LF band is read:

   ```python
   import spikeinterface.extractors as se
   from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc

   # SpikeGLX, with an LF file
   ap = se.read_spikeglx(raw_folder, stream_id='imec0.ap')
   lf = se.read_spikeglx(raw_folder, stream_id='imec0.lf')
   extract_si_qc(ap, out_folder, lf_recording=lf)
   ```

   For Open Ephys, give the stream names of the probe's AP and LFP streams, which
   `se.get_neo_streams('openephysbinary', raw_folder)` lists:

   ```python
   ap = se.read_openephys(raw_folder, stream_name='Record Node 101#Neuropix-PXI-100.ProbeA')
   lf = se.read_openephys(raw_folder, stream_name='Record Node 101#Neuropix-PXI-100.ProbeA-LFP')
   extract_si_qc(ap, out_folder, lf_recording=lf)
   ```

   Neuropixels 2.0 probes record wideband, without a separate LF band. Set `lfp_from_ap` to
   compute the LF band from the AP band, decimated to 2.5 kHz:

   ```python
   ap = se.read_cbin_ibl(cbin_file_path=raw_folder / 'recording.ap.cbin')
   extract_si_qc(ap, out_folder, lfp_from_ap=True)
   ```

   `lfp_from_ap` is refused for probes with a separate LF band, such as Neuropixels 1.0, whose
   AP band is high-passed and holds no LFP. With neither an LF recording nor `lfp_from_ap`, only
   the AP datasets are written. The
   recordings are processed as by `extract_ephys` (see {ref}`ref-input-datasets` for each file),
   reading a sample of the recording, so this takes minutes rather than a pass over the file.

3. Check that `out_folder` holds the `_iblqc_*` files, `_iblqc_ephysChannels.contactIds.npy` and
   `alignment_gui_raw_data_snippets.npy`.

4. Point the GUI at the analyzer and the extracted files. In a session YAML, give the analyzer as
   `spike_sorting` and `out_folder` as `processed_ephys`:

   ```yaml
   path: /path/to/session

   probes:
     probe00:
       datasets:
         spike_sorting:
           path: sorting_analyzer.zarr
         processed_ephys:
           path: extracted
         picks:
           path: picks
         output:
           path: alignment
   ```

   The analyzer is detected from its folder, so `format: spikeinterface` is only needed to make
   the format explicit (see {ref}`ref-session-yaml`). Without a YAML, put the analyzer, the
   extracted files and `xyz_picks.json` in one folder and open that folder: the analyzer is found
   inside it, as long as the folder holds only one.

## How the channels are matched

The extracted files hold every channel of the raw recording, while the analyzer may hold fewer,
for example when bad channels were removed before sorting. The GUI matches them by the probe
contact ids that probeinterface gives Neuropixels contacts (`e12`, or `s1e288` on multi-shank
probes), which `extract_si_qc` saves in `_iblqc_ephysChannels.contactIds.npy`. These ids are the
same whichever reader opened the recording. If the analyzer holds a contact the raw recording
does not, the session does not load, as the files were written from another recording.

An analyzer without contact ids is matched by its SpikeGLX channel ids (`imec0.ap#AP12`), or else
taken to hold the channels of the raw recording in order, which shows the wrong channels if
channels were removed before sorting.

## Other options

SpikeInterface's [`export_to_ibl_gui`](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/exporters/to_ibl.py)
converts the analyzer to ALF files instead, with its own RMS and power spectrum computed from the
analyzer's recording. Give its output folder as both `spike_sorting` and `processed_ephys`. It
does not write the raw data snippets.

## Related

- {ref}`ref-input-datasets`
- {ref}`howto-shanks-sorted-separately`
