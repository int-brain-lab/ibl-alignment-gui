# IBL Alignment GUI

A PyQt GUI for aligning electrophysiology recordings using anatomical landmarks. Electrophysiology
features recorded along a probe are aligned to the brain regions along its histology track, to
locate each recording channel in the brain.

New to the GUI? Start with {doc}`installation`, then the first tutorial.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step lessons on the sample data, from your first alignment to preparing your own
recordings.
:::

:::{grid-item-card} How-to guides
:link: how-to/index
:link-type: doc

Recipes for specific tasks with your own data: converting recordings, writing session files,
running the region prediction models.
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

How alignment works, which landmarks to trust, coordinate systems and the prediction models.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Datasets, session YAML keys, keyboard shortcuts, menus, command-line options and the Python API.
:::
::::

```{toctree}
:hidden:
:caption: Get started

installation
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: How-to guides

how-to/index
```

```{toctree}
:hidden:
:caption: Explanation

explanation/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
api
changelog
```

```{toctree}
:hidden:
:caption: Contributing

contributing/index
```
