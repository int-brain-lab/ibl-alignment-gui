(howto-convert-ccf-coordinates)=
# How to convert between CCF and IBL coordinates

This guide shows you how to convert points between the Allen CCF and the IBL coordinates relative to
bregma, for example to turn the output of a tracing tool into `xyz_picks.json`.

## Prerequisites

- `iblatlas`, which is installed with the GUI.

## Steps

1. Create the atlas:

   ```python
   import numpy as np
   from iblatlas.atlas import AllenAtlas

   atlas = AllenAtlas(25)
   ```

2. Check the units and axis order of your points. `ccf2xyz` takes CCF points in **µm** and returns
   IBL points in **m**; `xyz2ccf` takes IBL points in **m** and returns CCF points in **µm**. Both
   take a `ccf_order`: `'mlapdv'` or `'apdvml'`, the order of the axes of the CCF points.

3. Convert:

   **CCF to IBL**

   ```python
   ccf_um = np.array([[3000, 4000, 3000], [6000, 6000, 500]], dtype=float)  # ML, AP, DV
   xyz_m = atlas.ccf2xyz(ccf_um, ccf_order='mlapdv')

   ccf_um = np.array([[3000, 4000, 3000], [6000, 6000, 500]], dtype=float)  # AP, DV, ML
   xyz_m = atlas.ccf2xyz(ccf_um, ccf_order='apdvml')
   ```

   **IBL to CCF**

   ```python
   xyz_m = np.array([[2000, 4000, 0], [4000, -1000, -4000]]) / 1e6   # ML, AP, DV in m

   ccf_um = atlas.xyz2ccf(xyz_m, ccf_order='mlapdv')   # returned as ML, AP, DV
   ccf_um = atlas.xyz2ccf(xyz_m, ccf_order='apdvml')   # returned as AP, DV, ML
   ```

   The IBL points are always ordered ML, AP, DV.

4. For `xyz_picks.json`, multiply the IBL points by `1e6` to give µm, and save them under the key
   `xyz_picks` (see {ref}`howto-export-track-brainreg` for an example).

## Related

- {ref}`explanation-coordinate-systems`: the two coordinate systems and why they differ
