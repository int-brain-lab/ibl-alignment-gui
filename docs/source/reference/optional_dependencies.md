(ref-optional-dependencies)=
# Optional dependencies

This page lists the optional extras and what each one enables. The base install runs the GUI on data
on your own disk. Some features need extra packages, installed as extras:

```console
pip install -e ".[ibl]"
pip install -e ".[ephysatlas]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `ibl` | `ibllib` | Converting Kilosort output with `extract_data`; the `alignment-gui-ibl` command |
| `ephysatlas` | `ibleatools`, which also installs `ibllib`, `torch` and `xgboost` | The Channel Prediction plugin |

Installing `ephysatlas` therefore also gives everything the `ibl` extra provides.

## What changes without them

Without `ibl`
: `alignment-gui-ibl` exits with a message saying how to install the extra.
  `ibl_alignment_gui.convertors.extract_ephys` cannot be imported. Everything in `alignment-gui`
  works.

Without `ephysatlas`
: The `Channel Prediction` menu is not added to `Plugins`, and the predicted regions and Ephys
  Atlas feature plots are not offered.

Without `torch`
: Only relevant if `ibleatools` was installed without its dependencies. The Channel Prediction
  menu then has no entries for the ProbeTransformer and alignment models; the XGBoost decoder
  still works.
