(ref-interface)=
# Interface

This page describes the main window and what each part of it shows. Every menu entry is listed in
{ref}`ref-menus`, and keyboard shortcuts in {ref}`ref-shortcuts`.

```{image} ../_static/alignment_gui_layout.png
:alt: Layout of the alignment GUI, with the Ephys, Histology, Slice and Fit figures
:align: center
```

The window holds four figures. The data source button (`...`) in the top right corner opens a
session (see {ref}`howto-open-session`), and the alignment dropdown next to it chooses the saved
alignment the active shank starts from (see {ref}`howto-resume-alignment`).

Zoom into the Ephys, Histology and Slice figures with the mouse wheel for a closer look. Press
{kbd}`Shift+A` to reset the axes of every figure.

(ref-interface-ephys)=
## Ephys figure

Electrophysiology data along the probe, in three panels:

Image plots
: 2D data, as images or scatter plots. Switched from the `Image Plots` menu.

Line plots
: 1D data, as a line along the depth of the probe. Switched from the `Line Plots` menu.

Probe plots
: 2D data drawn on the probe geometry. Switched from the `Probe Plots` menu.

Prediction panel
: A channel prediction from the Channel Prediction plugin, shown after the probe plot (or after
  the feature plot, in the feature view) once one is chosen from the `Prediction Plots` menu.
  Predictions are placed by depth along the probe, like the other ephys plots. The panel carries
  the reference lines, and hovering over a predicted region adds its acronym to the panel's
  title. Choosing `None` removes it.

`All predictions (side by side)` ({kbd}`Shift+P`) replaces the image, line and probe plots with
one panel per prediction, to compare the models. Choosing any other plot returns to the usual view.

By default every unit from the spike sorting is shown. The `Filter Plots` menu restricts the plots
to a class of units; which filters are available depends on the fields in `clusters.metrics` (see
{ref}`ref-input-datasets`).

(ref-interface-histology)=
## Histology figure

The brain regions along the traced probe track, in three panels from left to right:

Scaled brain regions
: The regions after the current alignment has been applied.

Scale factor
: A heatmap of the scaling applied to each region along the track. Hovering over a region shows its
  scale factor in the title of the colour bar above the figure.

Original brain regions
: The regions before any alignment.

Black dotted lines in the left and right panels mark the deepest and the shallowest channel.

The regions in the right panel can be shown with three mappings, chosen from the `Mapping` menu
(see {ref}`explanation-region-mappings`):

Allen
: The full Allen atlas hierarchy.

Beryl
: Allen regions grouped into larger areas.

Cosmos
: The coarsest grouping, into a small number of major areas.

The Channel Prediction plugin adds its predicted regions to the same menu (see
{ref}`ref-channel-prediction`).

## Slice figure

A slice through the brain along the traced trajectory. Each row of the image is taken from the
trajectory coordinate at that depth, so a trajectory that is not smooth makes the image appear to
jump between slices.

Overlaid on the slice:

- a black line through the traced points, extended to the top and bottom of the brain
- the channels along the trajectory, coloured by the plot shown in the Probe plots panel
- the reference lines, as black dotted lines perpendicular to the trajectory

The intensity of the image is set with the scale bar on the right of the figure.

The image is chosen from the `Slice Plots` menu:

- `CCF`: the Allen atlas average template (always available)
- `Annotation`: the Allen atlas annotation (always available)
- `Histology red` and `Histology green`: one entry per histology volume found for the session
  (see {ref}`ref-input-datasets`)

(ref-interface-fit)=
## Fit figure

The scaling applied along the depth of the trajectory.

Fit line
: The piecewise fit along the trajectory. The points mark the reference lines.

Linear fit line (dotted)
: A linear fit through all the reference lines. Shown when there are three or more.

The `Linear fit` checkbox in the top left corner of the figure sets whether the regions beyond the
outermost reference lines are scaled by the linear fit. When it is unticked they are only offset.

With more than one shank in the grid layout, the Fit figure shows the fit of every shank.

## Buttons

Fit
: Apply the fit ({kbd}`Enter`).

Next / Previous
: Move through the last 10 fits ({kbd}`Shift+Right` / {kbd}`Shift+Left`).

Reset
: Return the active shank to the alignment it was loaded with ({kbd}`Shift+R`).

Upload
: Save the alignment (see {ref}`ref-output-files`) ({kbd}`Shift+U`).

Save Progress
: Save the alignment without uploading it, to recover it later ({kbd}`Shift+S`).

## Multiple shanks

When a session has more than one shank, or more than one configuration:

- In the tabbed layout each shank has its own tab. In the grid layout all shanks are shown at
  once. Switch between them with {kbd}`T`.
- Change the active shank with {kbd}`Left` and {kbd}`Right`, or by clicking its tab in the tabbed
  layout.
- The active shank's figure titles are shown in red.
- Uploading asks which shanks to upload.
