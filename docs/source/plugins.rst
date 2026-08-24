Plugins
=======

The IBL Alignment GUI supports a plugin architecture that allows users to extend the GUI with additional functionality. Plugins can be used to add new visualizations, interactive tools, or auxiliary views that integrate with the main GUI.

This section describes the currently available plugins and how to use them. The Channel
Prediction plugin, which predicts the brain region of each channel from its electrophysiology
features, has its own page: see :doc:`channel_prediction`.

Cluster Plugin
--------------

The Cluster plugin allows users to inspect individual spike-sorted clusters by visualizing their waveforms and inter-spike interval (ISI) distributions.

The plugin is attached to image plots that display cluster-level data.

Usage
~~~~~

To use the Cluster plugin:

1. Ensure an image plot displaying cluster-level data is active
2. Click on an individual cluster in the image plot

A new window will open showing:

- The waveform of the selected cluster
- The ISI distribution for that cluster

**Keyboard Shortcuts**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Shortcut
     - Action
   * - :kbd:`M`
     - Minimise / show the plugin windows
   * - :kbd:`Alt+X`
     - Close the plugin windows


Range Controller
----------------

The Range Controller plugin allows users to interactively adjust the data range displayed in different plot types.

The effect of the range controller depends on the plot type:

- **Image plots and probe plots**: Adjusts the color scale range
- **Line plots**: Adjusts the x-axis range

Usage
~~~~~

The Range Controller can be launched from the ``Plugins`` menu bar.

When opened, a new window appears with sliders controlling the minimum and maximum range values for each plot type.

When working with multi-shank or dual-configuration data, a radio button selector allows you to choose which shank or configuration the range controls apply to.

**Adjusting Ranges**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Action
     - Method
   * - Adjust range
     - Drag the min/max sliders for the desired plot type
   * - Set an exact range
     - Type the value into the box beside the corresponding slider
   * - Reset single plot range
     - Click **Reset** button next to the corresponding slider
   * - Reset all plot ranges
     - Press :kbd:`R`


3D Brain Viewer
---------------

The 3D Brain Viewer plugin provides a three-dimensional visualization of the probe trajectory within a brain model.

Usage
~~~~~

Launch the 3D Brain Viewer from the ``Plugins`` menu bar.

A new window will open displaying:

- A 3D brain model
- The probe trajectory overlaid in 3D space

The feature data shown along the probe automatically updates as you change the probe plots in the main GUI.

If a cluster plot is selected in the image plot, the corresponding cluster locations will be displayed in the 3D brain viewer.

**Display Controls**

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Control
     - Function
   * - Cluster marker size slider
     - Adjust size of cluster markers
   * - **Show regions** checkbox
     - Show / hide anatomical brain regions
   * - **Show picks** checkbox
     - Show / hide probe trajectory picks

Adding Additional Plots
-----------------------

Custom plots can be added to the GUI via the plugin system, without modifying the GUI itself.

``ibl_alignment_gui/plugins/allen/additional_plots_allen.py`` is a worked example. It shows how a
plugin adds its own plots by attaching new plot methods to the plot loader, so that they appear in
the menu bars alongside the built-in plots. ``ibl_alignment_gui/plugins/allen/custom_filters.py``
does the same for the unit filters, adding new entries to the ``Filter Plots`` menu.

.. note::
   Detailed developer documentation for writing custom plugins and additional plots will be provided in a future release.
