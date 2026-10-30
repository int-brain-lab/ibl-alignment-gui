(tutorial-preparing-your-data)=
# Preparing your data for the GUI

In this tutorial you will take a recording you have spike sorted with
[SpikeInterface](https://spikeinterface.readthedocs.io/), histology you have registered with
[brainreg](https://brainglobe.info/documentation/brainreg/index.html), and a probe track you have
identified with
[brainreg-segment](https://brainglobe.info/tutorials/segmenting-1d-tracks.html), and turn them
into a session the GUI can open. You will convert the spike sorting and the probe track, put the
histology images where the GUI can find them, write a session YAML that ties everything together,
and launch the GUI with it.

## Before you start

- Complete {ref}`tutorial-first-alignment`.
- Have these from your own experiment:

  | What | From |
  |---|---|
  | A `SortingAnalyzer` saved to a folder, with its recording attached | SpikeInterface |
  | The LF band of the recording, for example the SpikeGLX `.lf.bin` and `.lf.meta` files | Your recording system |
  | A brainreg output folder, registered to the 25 µm Allen mouse atlas (`allen_mouse_25um`) | brainreg |
  | The probe track, identified in atlas space with brainreg-segment | brainreg-segment |

- Have SpikeInterface installed in its own virtual environment, separate from the GUI's, so the
  two do not interfere:

  ```console
  uv venv si-env --python 3.12
  source si-env/bin/activate
  uv pip install "spikeinterface[full]"
  ```

  On Windows, activate it with `.\si-env\Scripts\activate` instead.

## 1. Create a folder for the session

Create a folder called `my_session`, with a subfolder for each kind of data, and open a terminal
in the folder that contains it:

```console
mkdir -p my_session/spike_sorting my_session/picks my_session/histology my_session/alignment
```

| Folder | Will hold |
|---|---|
| `spike_sorting` | The spike sorting, RMS and power spectrum files |
| `picks` | The traced probe track |
| `histology` | The registered histology images |
| `alignment` | The results the GUI writes |

## 2. Convert the spike sorting

SpikeInterface writes the files the GUI reads with `export_to_ibl_gui`.

1. With `si-env` active, save this as `export_spike_sorting.py`, with the two paths at the top
   set to your own:

   ```python
   from pathlib import Path

   import spikeinterface.core as si
   import spikeinterface.extractors as se
   from spikeinterface.exporters import export_to_ibl_gui

   analyzer_folder = Path('/path/to/sorting_analyzer')
   spikeglx_folder = Path('/path/to/spikeglx/recording')
   out_path = Path('my_session/spike_sorting')

   analyzer = si.load_sorting_analyzer(analyzer_folder)

   # The exporter needs the templates, amplitudes and quality metrics; compute any that are missing
   for extension in ['random_spikes', 'waveforms', 'templates', 'noise_levels',
                     'spike_amplitudes', 'quality_metrics']:
       if not analyzer.has_extension(extension):
           analyzer.compute(extension)

   # The LF band, used for the LF RMS and power spectrum plots
   lfp = se.read_spikeglx(spikeglx_folder, stream_name='imec0.lf')

   export_to_ibl_gui(analyzer, output_folder=out_path, lfp_recording=lfp, remove_if_exists=True)
   ```

   If your recording is not from SpikeGLX, read it with the matching SpikeInterface extractor
   instead.

2. Run it:

   ```console
   python export_spike_sorting.py
   ```

   It prints its progress as it computes the RMS and power spectra.

3. Look in `my_session/spike_sorting`. It holds:

   | Files | Used for |
   |---|---|
   | `spikes.*`, `clusters.*` | The spike sorting plots |
   | `channels.*` | The position of each channel on the probe |
   | `_iblqc_ephysTimeRms*`, `_iblqc_ephysSpectralDensityLF.*` | The RMS and LF power plots |

## 3. Convert the probe track

brainreg-segment saves each track you identify in atlas space as a `.npy` file in the
`segmentation/atlas_space/tracks` folder of your brainreg output. Its points are in Allen CCF
coordinates, in µm. The GUI needs them relative to bregma, as `xyz_picks.json`.

1. Activate the GUI's virtual environment, which has `iblatlas`.
2. Save this as `convert_track.py`, with the track path set to your own:

   ```python
   import json
   from pathlib import Path

   import numpy as np
   from iblatlas.atlas import AllenAtlas

   track_path = Path('/path/to/brainreg/output/segmentation/atlas_space/tracks/track_0.npy')
   picks_path = Path('my_session/picks')

   atlas = AllenAtlas(25)

   # brainreg-segment track: Allen CCF coordinates in µm, ordered AP, DV, ML
   xyz_apdvml = np.load(track_path)

   # IBL coordinates: ML, AP, DV relative to bregma, converted from m to µm
   xyz_mlapdv = atlas.ccf2xyz(xyz_apdvml, ccf_order='apdvml') * 1e6

   with open(picks_path / 'xyz_picks.json', 'w') as f:
       json.dump({'xyz_picks': xyz_mlapdv.tolist()}, f, indent=2)

   regions = atlas.regions.get(atlas.get_labels(xyz_mlapdv / 1e6)).acronym
   print('Regions along the track:', list(dict.fromkeys(regions)))
   ```

3. Run it:

   ```console
   python convert_track.py
   ```

   It prints the brain regions the track passes through, from the entry point down. Check that
   they are the ones you expect for this probe. If they are not, the track was probably traced in
   sample space rather than atlas space.

The conversion is explained in {ref}`explanation-coordinate-systems`.

## 4. Add the histology images

brainreg writes each of your histology images, warped into the atlas, as a TIFF file called
`downsampled_standard_<IMAGE_NAME>.tiff`, where `<IMAGE_NAME>` is the name of that image. These
are the images the Slice figure can show.

Copy them into `my_session/histology`, renaming them so the GUI can tell the channels apart: `GR`
in the name for one channel and `RD` for the other. For example:

```console
cp /path/to/brainreg/output/downsampled_standard_<IMAGE_NAME_1>.tiff my_session/histology/histology_GR.tiff
cp /path/to/brainreg/output/downsampled_standard_<IMAGE_NAME_2>.tiff my_session/histology/histology_RD.tiff
```

The GUI loads at most one `GR` and one `RD` image. Which is which only sets their names in the
`Slice Plots` menu (see {ref}`howto-use-histology-images`).

## 5. Write the session YAML

Save this as `my_session/session.yaml`, with `path` set to the full path of `my_session` and
`raw_ephys` set to your SpikeGLX folder:

```yaml
path: /full/path/to/my_session

probes:
  probe00:
    datasets:
      spike_sorting:
        path: spike_sorting
      processed_ephys:
        path: spike_sorting
      raw_ephys:
        path: /path/to/spikeglx/recording
      picks:
        path: picks
      histology:
        path: histology
      output:
        path: alignment
```

`export_to_ibl_gui` writes the spike sorting and the RMS and power spectrum files to the same
folder, so `spike_sorting` and `processed_ephys` both point at it. `raw_ephys` is optional: with
it, the GUI also shows snippets of the raw signal. Every key is described in
{ref}`ref-session-yaml`.

## 6. Launch the GUI

1. With the GUI's environment active, run:

   ```console
   alignment-gui -y my_session/session.yaml
   ```

2. Check that everything has loaded:

   - `Image Plots` lists the spike sorting plots, such as `Firing Rate` and `Amplitude`, and the
     RMS and LF plots
   - `Probe Plots` lists the RMS plots and the five LF power bands
   - the Histology figure shows the regions along your traced track, the same ones
     `convert_track.py` printed
   - `Slice Plots` lists `Histology red` and `Histology green`, and with either of them chosen
     ({kbd}`Alt+4`) the Slice figure shows your histology with the track drawn on it

Your data is now ready to align, in the same way as in {ref}`tutorial-first-alignment`.

## What you've learned

You have:

- exported a SpikeInterface sorting to the files the GUI reads
- converted a probe track identified with brainreg-segment to `xyz_picks.json`
- added your registered histology so it shows in the Slice figure
- written a session YAML and launched the GUI with it

## Next steps

- {ref}`howto-prepare-data`: other ways of preparing your data, such as from kilosort or Lasagna
- {ref}`howto-configure`: session YAMLs for multi-shank probes and other layouts
- {ref}`ref-input-datasets`: every file the GUI reads
