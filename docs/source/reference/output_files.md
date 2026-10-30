(ref-output-files)=
# Output files

This page lists the files the GUI writes. They go in the `output` folder of the session YAML.

## Files

`channel_locations.json`
: Written when an alignment is uploaded. The location of every channel after the alignment, keyed
  `channel_0`, `channel_1`, …, plus an `origin` entry.

  | Field | Units | Description |
  |---|---|---|
  | `x`, `y`, `z` | µm | ML, AP and DV coordinates relative to bregma |
  | `axial` | µm | Position along the probe (see {ref}`ref-probe-geometry`) |
  | `lateral` | µm | Position across the probe |
  | `brain_region_id` | | Allen atlas region id |
  | `brain_region` | | Allen atlas region acronym |
  | `original_channel_idx` | index | Row of the channel in `channels.localCoordinates`, or of the electrode in the metadata if there is no spike sorting |

  The `origin` entry gives the position of bregma in the Allen CCF, in µm, as
  `{"bregma": [ML, AP, DV]}`.

  ```json
  {
    "channel_0": {
      "x": -2453.87, "y": -4189.74, "z": -3415.69,
      "axial": 20.0, "lateral": 43.0,
      "brain_region_id": 0, "brain_region": "void",
      "original_channel_idx": 0
    },
    "origin": {"bregma": [5739.0, 5400.0, 332.0]}
  }
  ```

`prev_alignments.json`
: Written when an alignment is uploaded. Every alignment made for the probe, so that any of them
  can be reloaded from the alignment dropdown. Each key is the date and time of the alignment,
  followed by `_<user>` when a user name is known. Each value is a pair of lists: the depths of
  the reference lines in the electrophysiology, and the matching depths along the histology track.

`alignment_progress.json`
: Written by **Save Progress** ({kbd}`Shift+S`). An alignment that has not been uploaded yet. The
  next time the session is loaded it is offered in the alignment dropdown under a `recovered` key
  and chosen as the starting alignment. Deleted when the alignment is uploaded.

## Multi-shank probes

For a probe with more than one shank, each shank is aligned separately and every file gets a
shank suffix, counting from 1 with no separator before the number:

```text
channel_locations_shank1.json
prev_alignments_shank1.json
alignment_progress_shank1.json
```

The suffix is only added when the shanks come from one probe entry in the session YAML. When each
shank is given as its own probe entry, each entry writes files without a suffix, so each entry
needs its own `output` path.
