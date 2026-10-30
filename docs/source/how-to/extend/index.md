(howto-extend)=
# Extend the GUI

These guides show you how to add your own plots and unit filters to the GUI with a plugin. A plugin
is a Python module with a `setup` function that hooks your functions into the GUI. It is
registered in `ibl_alignment_gui/plugins/add_plugins.py`. The built-in plugins are listed in
{ref}`ref-plugins`.

You need a clone of the repository installed in editable mode (`pip install -e .`), so that the
GUI picks up your changes.

The guides cover:

Adding a plot
: Add your own image, line or probe plot, computed from the data the GUI has loaded.

Adding a unit filter
: Add an entry to the `Filter Plots` menu that shows only the units matching your own criterion.

```{toctree}
:maxdepth: 1

add_plot
add_unit_filter
```
