(howto-export-track-lasagna)=
# How to export a track from Lasagna

This guide shows you how to turn a track traced with
[Lasagna](https://github.com/SainsburyWellcomeCentre/lasagna/wiki/Use-case:-tracing-electrode-tracks)
into `xyz_picks.json`.

## Prerequisites

- The `ibl` extra installed (see {ref}`howto-install-extras`).

## Steps

1. Trace the track in Lasagna, following its guide, and:

   - do not rotate, flip or mirror the images while tracing
   - trace in histology already registered to the Allen atlas, or apply the registration transform
     to the points afterwards if you trace in the original image space

2. Save the `_pts` line, not the `_fit` line.

3. Convert it to IBL coordinates and save it:

   ```python
   import json
   from pathlib import Path

   from ibllib.pipes.histology import load_track_csv

   picks_path = Path('/path/to/picks')

   # Returns ML, AP, DV relative to bregma in m; converted to µm
   xyz = load_track_csv('/path/to/lasagna/tracing_pts.csv') * 1e6

   with open(picks_path / 'xyz_picks.json', 'w') as f:
       json.dump({'xyz_picks': xyz.tolist()}, f, indent=2)
   ```

4. In the session YAML, use `picks_path` as the `picks` folder (see
   {ref}`howto-single-probe-yaml`).

## Related

- {ref}`explanation-coordinate-systems`
