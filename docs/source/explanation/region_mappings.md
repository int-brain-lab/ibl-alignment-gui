(explanation-region-mappings)=
# Brain region mappings

The Histology figure can show the brain regions along the track at three levels of detail, chosen
from the `Set Mapping` menu. This page explains what they are and when each is
useful.

## The three mappings

Allen
: The full hierarchy of the Allen atlas annotation. Every voxel is labelled with its most specific
  region, including cortical layers and small nuclei.

Beryl
: Allen regions grouped into larger areas, used across IBL analyses. Cortical layers are merged
  into their cortical area, and fine subdivisions into their parent region.

Cosmos
: The coarsest grouping, into about ten major divisions of the brain, such as isocortex,
  hippocampal formation, thalamus and midbrain.

All three are views of the same annotation. Changing the mapping changes how regions are grouped
and coloured, not where the channels are. The saved `channel_locations` always use the Allen
regions.

## Choosing a mapping while aligning

Coarse mappings make the large boundaries easy to see: the edge of the brain, the boundary between
cortex and hippocampus, or between thalamus and midbrain. These are often the most reliable
landmarks, so a common approach is to place the first reference lines with Cosmos or Beryl, and
switch to Allen to refine the alignment within a structure, for example across cortical layers.

Fine Allen regions can be thinner than the spacing between channels, or than the uncertainty in
the traced track. Matching a feature in the electrophysiology to such a region is less reliable
than matching it to a large boundary.
