(howto-export-track-brainreg)=
# How to export a track from brainreg-segment

This guide shows you how to turn a track traced with
[brainreg-segment](https://brainglobe.info/tutorials/segmenting-1d-tracks.html) into
`xyz_picks.json`.

## Prerequisites

- The histology registered to the Allen atlas with brainreg.
- `iblatlas`, which is installed with the GUI.

## Steps

1. Trace the track in brainreg-segment, following its tutorial, with these settings:

   | Setting | Value |
   |---|---|
   | Tracing space | Registered atlas space, not the original sample space |
   | Spline points | Fewer than 100 when fitting the track |

2. Click **To brainrender** to save the track as a `.npy` file. It is written to the
   `segmentation/atlas_space/tracks` folder of the brainreg output, named after the track.

3. Convert it to IBL coordinates and save it:

   ```python
   import json
   from pathlib import Path

   import numpy as np
   from iblatlas.atlas import AllenAtlas

   atlas = AllenAtlas(25)
   picks_path = Path('/path/to/picks')

   # brainreg track: CCF coordinates in µm, ordered AP, DV, ML
   xyz_apdvml = np.load('/path/to/brainreg/output/segmentation/atlas_space/tracks/track_0.npy')

   # IBL coordinates: ML, AP, DV relative to bregma, converted from m to µm
   xyz_mlapdv = atlas.ccf2xyz(xyz_apdvml, ccf_order='apdvml') * 1e6

   with open(picks_path / 'xyz_picks.json', 'w') as f:
       json.dump({'xyz_picks': xyz_mlapdv.tolist()}, f, indent=2)
   ```

4. For a multi-shank probe, trace each shank and save each as `xyz_picks_shank<N>.json`, counting
   from 1.

5. In the session YAML, use `picks_path` as the `picks` folder (see
   {ref}`howto-single-probe-yaml`).

## Related

- {ref}`explanation-coordinate-systems`
- {ref}`howto-use-histology-images`: show the brainreg images in the Slice figure
