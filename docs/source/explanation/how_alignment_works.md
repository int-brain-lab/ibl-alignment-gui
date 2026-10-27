(explanation-how-alignment-works)=
# How alignment works

## Why an alignment is needed

The traced probe track gives the path of the probe through the brain, but not exactly where along
that path each channel sits. The two don't line up, for several reasons:

- the brain shrinks and deforms when it is fixed, sliced and imaged, so distances measured in the
  histology are not the distances in the living brain
- the track is traced by hand from images and registered to an atlas, and both steps add error
- the tip of the probe is not always visible, so the depth the probe reached is uncertain

The electrophysiology, on the other hand, is recorded at known positions along the probe. Features
in it, such as a band of high LFP power or a gap in firing where the probe crosses white matter,
mark boundaries between brain regions. Matching those features to the same boundaries in the
histology tells the GUI how to stretch and shift the track so that the channels land in the right
regions.

## Two depth axes

The GUI works with two one-dimensional axes:

Feature depth
: Depth along the probe, as shown in the {ref}`Ephys figure <ref-interface-ephys>`. Channel
  positions are fixed on this axis.

Track depth
: Distance along the traced trajectory through the brain, as shown in the
  {ref}`Histology figure <ref-interface-histology>`.

Feature depth 0 is the deepest channel. Track depth 0 is 200 µm above the deepest traced point,
which is taken as the probe tip. An alignment is a mapping from feature depth to track depth.
Before any alignment the mapping is the identity: the deepest channel sits 200 µm above the end of
the traced track, and a channel 1 mm above it sits 1 mm further along the track.

## Reference line pairs

A reference line pair is a claim that a point in the electrophysiology (one line, in the Ephys
figure) and a point along the track (the other line, in the Histology figure) are the same place
in the brain. Each fit maps the feature depth of every reference line onto the track depth of its
partner, and maps the depths between the lines by linear interpolation.

What happens beyond the outermost lines depends on how many pairs there are:

One pair
: Everything is shifted by the same amount. The track is not stretched anywhere.

Two pairs
: The track between the two lines is stretched or compressed uniformly so that both pairs match.
  Outside the lines, the track is only shifted, using the offset of the nearest line.

Three or more pairs
: Each stretch between neighbouring lines is scaled separately, so the mapping is piecewise
  linear. Beyond the outermost lines, the scaling follows a straight line fitted through all the
  pairs (shown dotted in the {ref}`Fit figure <ref-interface-fit>`). Unticking `Linear fit` in the
  Fit figure makes the outer parts shift only, as with two pairs.

With three or more pairs, the straight-line fit gives the regions beyond the lines one overall
scaling, taken from all the pairs, rather than carrying the local stretch of the outermost
segment over the rest of the probe.

## Fits build on each other

Each time a fit is applied, the reference lines in the Histology figure are read in the current,
already-aligned view, and converted back to the original track through the previous fit. The new
fit therefore refines the previous one rather than starting again. The last 10 fits are kept, so
**Previous** and **Next** can step between them; **Reset** returns to the alignment the session was
loaded with.

## Scale factors

For each brain region along the track, the scale factor is its length after the alignment divided
by its length in the original histology. A factor above 1 means the region was stretched to match
the electrophysiology, below 1 that it was compressed. Neighbouring regions with the same factor are
shown as one block in the scale factor panel.

Large or very different scale factors between neighbouring segments are worth checking: they can
mean that a reference line pair matches a feature to the wrong boundary.

## What is saved

When an alignment is uploaded, the feature and track depths of the reference lines are saved, so
the alignment can be reloaded and refined later, together with the resulting 3D position and brain
region of every channel. The file formats are in {ref}`ref-output-files`.
