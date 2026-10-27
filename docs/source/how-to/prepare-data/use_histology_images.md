(howto-use-histology-images)=
# How to show your own histology in the Slice figure

This guide shows you how to display your registered histology images behind the trajectory, instead
of only the Allen template and annotation.

## Prerequisites

- Histology registered to the 25 µm Allen atlas, saved as NRRD or TIFF volumes with these
  dimensions:

  | Format | Dimensions |
  |---|---|
  | NRRD | [voxels in AP, voxels in DV, voxels in ML] |
  | TIFF (`.tif` or `.tiff`) | [voxels in ML, voxels in DV, voxels in AP] |

## Steps

1. Put the volumes in one folder. Name them so the GUI can tell the channels apart:

   | Format | Red channel | Green channel |
   |---|---|---|
   | NRRD | `RD` in the file name | `GR` in the file name |
   | TIFF | `RD` or `C1` in the file name | `GR` or `C0` in the file name |

   For TIFFs, if you don't want to name the channels by colour, call them `C0` and `C1` instead.
   The name only sets the entry in the `Slice Plots` menu: `C0` appears as `Histology green` and
   `C1` as `Histology red`.

2. Add the folder as the `histology` dataset in the session YAML. Histology is usually shared by
   all the probes of an animal, so it is convenient to put it in `defaults`:

   ```yaml
   defaults:
     histology:
       path: /path/to/histology
   ```

   When opening a data folder instead of a YAML, put the volumes in that folder.

3. Load the session and cycle through the slice images with {kbd}`Alt+4`. Each volume adds a
   `Histology red` or `Histology green` entry.

If both NRRD and TIFF files are in the folder, only the NRRD files are used.

## Related

- {ref}`ref-input-datasets`: histology file matching
