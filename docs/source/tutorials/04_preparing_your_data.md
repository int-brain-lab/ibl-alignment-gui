(tutorial-preparing-your-data)=
# Preparing your data for the GUI

In this tutorial you will take a recording you have spike sorted with
[SpikeInterface](https://spikeinterface.readthedocs.io/), histology you have registered with
[brainreg](https://brainglobe.info/documentation/brainreg/index.html), and a probe track you have
identified with
[brainreg-segment](https://brainglobe.info/tutorials/segmenting-1d-tracks.html), and turn them
into a session the GUI can open. The GUI reads your `SortingAnalyzer` as it is; you will extract
the plots it needs from the raw recording, convert the probe track, put the histology images
where the GUI can find them, write a session YAML that ties everything together, and launch the
GUI with it.

## Before you start

- Complete {ref}`tutorial-first-alignment`.
- Have these from your own experiment:

  | What | From |
  |---|---|
  | A `SortingAnalyzer` saved to a folder or a `.zarr` | SpikeInterface |
  | The raw recording the analyzer was sorted from, for example the SpikeGLX `.ap.bin`, `.lf.bin` and `.meta` files | Your recording system |
  | A brainreg output folder, registered to the 25 µm Allen mouse atlas (`allen_mouse_25um`) | brainreg |
  | The probe track, identified in atlas space with brainreg-segment | brainreg-segment |

- Install the GUI's `spikeinterface` extra, so the GUI can read the analyzer (see
  {ref}`howto-install-extras`). With the GUI's environment active, in the folder you cloned it
  into:

  ```console
  uv pip install -e ".[spikeinterface]"
  ```

  Every step of this tutorial runs in this one environment. It must be able to load your
  analyzer: if you sorted with a newer version of SpikeInterface than it has, upgrade it with
  `uv pip install -U spikeinterface`.

## 1. Create a folder for the session

Create a folder called `my_session`, with a subfolder for each kind of data, and open a terminal
in the folder that contains it:

```console
mkdir -p my_session/extracted my_session/picks my_session/histology my_session/alignment
```

| Folder | Will hold |
|---|---|
| `extracted` | The RMS, power spectrum and raw data snippet files |
| `picks` | The traced probe track |
| `histology` | The registered histology images |
| `alignment` | The results the GUI writes |

## 2. Prepare the spike sorting and raw data

The GUI reads the spikes, units and channels from the analyzer. The RMS, power spectrum and raw
data plots come from the raw recording, which `extract_si_qc` reads once and summarises in a few
small files.

1. Save this as `prepare_ephys.py`, with the two paths at the top set to your own:

   ```python
   from pathlib import Path

   import spikeinterface as si
   import spikeinterface.extractors as se

   from ibl_alignment_gui.convertors.extract_si_qc import extract_si_qc

   analyzer_folder = Path('/path/to/sorting_analyzer')
   spikeglx_folder = Path('/path/to/spikeglx/recording')
   out_path = Path('my_session/extracted')

   # The extensions the GUI reads; compute any that are missing
   analyzer = si.load(analyzer_folder)
   for extension in ['random_spikes', 'templates', 'spike_amplitudes',
                     'spike_locations', 'template_metrics', 'quality_metrics']:
       if not analyzer.has_extension(extension):
           analyzer.compute(extension)

   # The raw recording, as read from disk
   ap = se.read_spikeglx(spikeglx_folder, stream_id='imec0.ap')
   lf = se.read_spikeglx(spikeglx_folder, stream_id='imec0.lf')

   extract_si_qc(ap, out_path, lf_recording=lf)
   ```

   If your recording is not from SpikeGLX, or is from a Neuropixels 2.0 probe without an LF
   band, read it as shown in {ref}`howto-convert-spikeinterface`.

2. Run it:

   ```console
   python prepare_ephys.py
   ```

   It shows its progress as it computes the RMS and power spectra, which takes a few minutes.

3. Look in `my_session/extracted`. It holds:

   | Files | Used for |
   |---|---|
   | `_iblqc_ephysTimeRms*`, `_iblqc_ephysSpectralDensityLF.*` | The RMS and LF power plots |
   | `alignment_gui_raw_data_snippets.npy`, `alignment_gui_raw_lf_data_snippets.npy` | The raw data snippets and the bad channel plots |
   | `_iblqc_ephysChannels.contactIds.npy` | Matching the analyzer's channels to the raw recording's |

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
`spike_sorting` set to your analyzer:

```yaml
path: /full/path/to/my_session

probes:
  probe00:
    datasets:
      spike_sorting:
        path: /path/to/sorting_analyzer
      processed_ephys:
        path: extracted
      picks:
        path: picks
      histology:
        path: histology
      output:
        path: alignment
```

The GUI recognises the analyzer from its folder, and reads the extracted files, snippets
included, from `processed_ephys`, so the raw recording is not needed to open the session. Every
key is described in {ref}`ref-session-yaml`.

## 6. Launch the GUI

1. With the GUI's environment active, run:

   ```console
   alignment-gui -y my_session/session.yaml
   ```

2. Check that everything has loaded:

   - `Image Plots` lists the spike sorting plots, such as `Firing Rate` and `Amplitude`, the RMS
     and LF plots, and the `Raw ap snippet` and `Raw lf snippet` images
   - `Line Plots` lists `Dead Channels`, `Noisy Channels Coherence`, `Noisy Channels PSD` and
     `Outside Channels`
   - `Probe Plots` lists the RMS plots and the five LF power bands
   - the Histology figure shows the regions along your traced track, the same ones
     `convert_track.py` printed
   - `Slice Plots` lists `Histology red` and `Histology green`, and with either of them chosen
     ({kbd}`Alt+4`) the Slice figure shows your histology with the track drawn on it

Your data is now ready to align, in the same way as in {ref}`tutorial-first-alignment`.

## What you've learned

You have:

- extracted the RMS, power spectra and raw data snippets of a SpikeInterface recording, for
  the GUI to show beside its `SortingAnalyzer`
- converted a probe track identified with brainreg-segment to `xyz_picks.json`
- added your registered histology so it shows in the Slice figure
- written a session YAML and launched the GUI with it

SpikeInterface's `export_to_ibl_gui` is another route, which converts the analyzer to ALF files
instead (see {ref}`howto-convert-spikeinterface`).

## Next steps

- {ref}`howto-prepare-data`: other ways of preparing your data, such as from kilosort or Lasagna
- {ref}`howto-configure`: session YAMLs for multi-shank probes and other layouts
- {ref}`ref-input-datasets`: every file the GUI reads
