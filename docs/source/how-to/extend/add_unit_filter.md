(howto-add-unit-filter)=
# How to add your own unit filter

This guide shows you how to add an entry to the `Filter Plots` menu that shows only the units
matching your own criterion. The example keeps units firing at more than 5 spikes per second.

## How filters work

A filter is a function that takes the `clusters.metrics` table, one row per unit, and returns a
boolean array of the units to keep. Filters are registered by name in
`ibl_alignment_gui.loaders.plot_loader.CUSTOM_FILTERS`, which the `Filter Plots` menu is built from
when a session is loaded.

## Steps

1. Create `ibl_alignment_gui/plugins/my_filters.py`:

   ```python
   import numpy as np

   from ibl_alignment_gui.loaders import plot_loader
   from iblutil.util import Bunch

   PLUGIN_NAME = 'My Filters'


   def setup(controller):
       """Register the plugin, and the function that adds its filters when data is loaded."""
       controller.plugins[PLUGIN_NAME] = Bunch()
       controller.plugins[PLUGIN_NAME]['activated'] = True
       controller.plugins[PLUGIN_NAME]['load_data'] = add_filters


   def add_filters(_):
       """Add the filters to the Filter Plots menu."""
       plot_loader.CUSTOM_FILTERS['High firing rate'] = filter_high_firing_rate


   def filter_high_firing_rate(metrics):
       """Keep the units firing at more than 5 spikes per second."""
       return np.asarray(metrics['firing_rate'] > 5)
   ```

   Change `firing_rate` to a column that is in your `clusters.metrics` file.

2. Register the plugin in `ibl_alignment_gui/plugins/add_plugins.py`, as for a plot:

   ```python
   from ibl_alignment_gui.plugins.my_filters import setup as setup_my_filters

   class Plugins:
       def __init__(self, controller):
           ...
           setup_my_filters(controller)
   ```

3. Start the GUI and load a session. `High firing rate` is in the `Filter Plots` menu.

If the column is missing, or the function raises an error, a warning is logged and every unit is
shown.

## Related

- {ref}`howto-filter-units`: the built-in filters
- {ref}`howto-add-plot`
