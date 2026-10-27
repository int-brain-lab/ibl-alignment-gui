(howto-add-plot)=
# How to add your own plot

This guide shows you how to add a plot to the `Image Plots`, `Line Plots` or `Probe Plots` menu,
computed from the data the GUI has loaded. The example adds a line plot of the number of spikes
along the probe.

## How plots are found

Each shank has a plot loader, an instance of
{class}`~ibl_alignment_gui.loaders.plot_loader.PlotLoader`. When the plots are built, the loader
calls every one of its methods whose name starts with `image_`, `scatter_`, `line_`, `probe_` or
`feature_`. Each returns a dict of `{menu name: plot data}`, where the plot data is an
`ImageData`, `ScatterData`, `LineData` or `ProbeData` from the same module. A plugin adds a plot by
attaching such a method to each shank's loader, before the plots are built.

## Steps

1. Create `ibl_alignment_gui/plugins/my_plots.py`:

   ```python
   from types import MethodType

   import numpy as np

   from ibl_alignment_gui.loaders.plot_loader import LineData, skip_missing
   from ibl_alignment_gui.utils.helpers import shank_loop
   from iblutil.util import Bunch

   PLUGIN_NAME = 'My Plots'


   def setup(controller):
       """Register the plugin, and the function that adds its plots when data is loaded."""
       controller.plugins[PLUGIN_NAME] = Bunch()
       controller.plugins[PLUGIN_NAME]['activated'] = True
       controller.plugins[PLUGIN_NAME]['load_data'] = add_plots


   @shank_loop
   def add_plots(_, items, **kwargs):
       """Attach the plot methods to the plot loader of every shank."""
       plots = items.model.loaders['plots']
       plots.line_spike_count = MethodType(line_spike_count, plots)


   @skip_missing(['spikes'])
   def line_spike_count(self):
       """Line plot of the number of spikes in 20 µm bins along the probe."""
       bins = np.arange(self.chn_min, self.chn_max + 20, 20)
       counts, _ = np.histogram(self.spike_depths, bins)
       depths = bins[:-1] + 10

       line = LineData(
           x=counts,
           y=depths,
           xrange=np.array([0, np.max(counts)]),
           levels=np.array([0, np.max(counts)]),
           default_levels=np.array([0, np.max(counts)]),
           xaxis='Number of spikes',
       )
       return {'Spike count': line}
   ```

   - `load_data` is called each time a session is loaded, after the shanks are created and before
     the plots are built.
   - `shank_loop` calls `add_plots` once for every shank and configuration.
   - `skip_missing(['spikes'])` skips the plot when the session has no spike sorting.
   - Inside the method, `self` is the plot loader. `self.data` holds the loaded data, and
     properties such as `self.spike_depths` give the spikes of the units selected by the current
     filter, so the plot follows the `Filter Plots` menu.

2. Register the plugin in `ibl_alignment_gui/plugins/add_plugins.py`. Import its `setup`, and call
   it in `Plugins.__init__` with the other plugins:

   ```python
   from ibl_alignment_gui.plugins.my_plots import setup as setup_my_plots

   class Plugins:
       def __init__(self, controller):
           ...
           setup_my_plots(controller)
   ```

3. Start the GUI and load a session. `Spike count` is in the `Line Plots` menu.

For an image or probe plot, return an `ImageData` or `ProbeData` and start the method name with
`image_` or `probe_`. The built-in plots in `plot_loader.py` are worked examples of each.

## Related

- {ref}`howto-add-unit-filter`
- {doc}`../../api`: `ibl_alignment_gui.loaders.plot_loader`
