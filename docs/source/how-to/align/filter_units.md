(howto-filter-units)=
# How to show only well-isolated units

This guide shows you how to restrict the spike sorting plots to good units, so that landmarks are
not hidden by noise or multi-unit activity.

## Prerequisites

- A `clusters.metrics` file in the spike sorting folder, with a `ks2_label` column (for the
  kilosort filters) or a `label` column (for the IBL filter).

## Steps

1. Open the `Filter Plots` menu.

   ```{image} ../../_static/how_to/align/filter_plots.png
   :alt: The Filter Plots menu, with the All, KS good, KS mua and IBL good filters
   :align: center
   ```

2. Choose a filter:

   | Filter | Units shown | Needs |
   |---|---|---|
   | `All` | Every unit | |
   | `KS good` | Units kilosort labelled good | `ks2_label` |
   | `KS mua` | Units kilosort labelled multi-unit | `ks2_label` |
   | `IBL good` | Units passing the IBL quality metrics | `label` |

If the column a filter needs is missing, a warning is logged and every unit is shown.

## Related

- {ref}`ref-input-datasets`: `clusters.metrics`
