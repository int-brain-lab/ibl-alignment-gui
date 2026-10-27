(explanation-coordinate-systems)=
# Coordinate systems

The GUI reads and writes positions in the brain relative to bregma, while atlases and many
registration tools use the Allen Common Coordinate Framework (CCF), whose origin is a corner of the
atlas volume. This page explains the two and how they relate. The conversions themselves are in
{ref}`howto-convert-ccf-coordinates`.

## IBL coordinates

The traced trajectory (`xyz_picks`) and the channel locations (`channel_locations`) are given as
x, y, z in µm, relative to bregma:

| Axis | Direction | Positive towards |
|---|---|---|
| x | Medial–lateral (ML) | Right |
| y | Anterior–posterior (AP) | Anterior |
| z | Dorsal–ventral (DV) | Dorsal |

Because z is positive dorsally and bregma sits on the skull surface, points inside the brain have
negative z. For example, a point at `[-2000, -3000, -2500]` is 2 mm left of the midline, 3 mm
posterior of bregma and 2.5 mm below it.

Internally the GUI works in metres; files use µm.

## The Allen CCF

The Allen CCF is a 3D reference brain. Its coordinates are measured from the front, top, left
corner of the atlas volume, with axes in the order anterior–posterior, dorsal–ventral,
medial–lateral, each positive towards posterior, ventral and right. Tools that register histology
to the Allen atlas, such as brainreg, usually give positions in this frame.

## Bregma in the CCF

The two frames are related by the position of bregma in the CCF volume:

| Axis | Bregma in the CCF |
|---|---|
| ML | 5739 µm |
| AP | 5400 µm |
| DV | 332 µm |

Converting between the frames means subtracting (or adding) this position, reordering the axes,
and flipping the AP and DV directions. `iblatlas` does this with `AllenAtlas.ccf2xyz` and
`AllenAtlas.xyz2ccf`. Every `channel_locations` file records the bregma position it was computed
with, in its `origin` entry.

## Atlas resolution

The GUI uses the 25 µm Allen atlas. Histology images given to the GUI must be registered to it,
and region lookups are made at that resolution: features finer than 25 µm along the track cannot be
resolved by the atlas regions.
