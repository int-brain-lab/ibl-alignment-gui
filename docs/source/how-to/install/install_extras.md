(howto-install-extras)=
# How to install the optional extras

This guide shows you how to add the packages needed for converting kilosort output, the IBL online
mode or the Channel Prediction plugin.

## Prerequisites

- The GUI installed from a clone of the repository (see {doc}`../../installation`), with its
  virtual environment activated.

## Steps

1. Go to the folder you cloned the repository into:

   ```console
   cd ibl-alignment-gui
   ```

2. Install the extra you need. With `uv`:

   ```console
   uv pip install -e ".[ibl]"
   uv pip install -e ".[ephysatlas]"
   ```

   With `pip` (in a `venv` or `conda` environment):

   ```console
   pip install -e ".[ibl]"
   pip install -e ".[ephysatlas]"
   ```

   `ephysatlas` also installs everything `ibl` does, so for both, install `ephysatlas` only.

3. Check the extra is available:

   ```console
   python -c "import ibllib"       # ibl
   python -c "import ephysatlas"   # ephysatlas
   ```

   With `ephysatlas` installed, a `Channel Prediction` entry appears in the `Plugins` menu the
   next time the GUI starts.

## Related

- {ref}`ref-optional-dependencies`: what each extra enables
