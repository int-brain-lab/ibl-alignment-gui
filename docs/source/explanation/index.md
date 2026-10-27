(explanation)=
# Explanation

These pages explain how the GUI works and why it works that way. Unlike the tutorials and how-to
guides, they aren't step-by-step instructions. They cover the concepts the GUI is built on, such as
how a fit is calculated or how coordinates are defined. They're especially useful when an alignment
doesn't behave as you expect, or when you're deciding how to set up your own data.

The pages are grouped by topic:

The alignment itself
: {doc}`how_alignment_works`, from reference lines to a location for every channel.

Positions in the brain
: {doc}`coordinate_systems`, how bregma-relative and Allen CCF coordinates relate, and
  {doc}`region_mappings`, the three levels of detail the brain regions can be shown at.

Organising your data
: {doc}`shanks_probes_configurations`, {doc}`session_yaml_resolution`, and
  {doc}`offline_and_online_modes`, working from local files or from the IBL Alyx database.

The region prediction models
: {doc}`channel_prediction_models`, what the models predict, how to read their output, and their
  limits.

```{toctree}
:maxdepth: 1

how_alignment_works
coordinate_systems
region_mappings
shanks_probes_configurations
session_yaml_resolution
channel_prediction_models
offline_and_online_modes
```
