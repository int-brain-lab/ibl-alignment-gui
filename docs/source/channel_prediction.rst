Channel Prediction
==================

The Channel Prediction plugin predicts the brain region of each recording channel from its
electrophysiology features, so that the prediction can be compared against the histology while
aligning. It also makes the features themselves available as plots, so they can be inspected
alongside the other electrophysiology data.

Two independent models are provided:

**Inference model**
    A region classifier that predicts a region for each channel directly from its features.

**Spatial encoder**
    A model that predicts the features expected along the probe track and warps the recording onto
    that prediction, giving a region for each channel and an estimate of the alignment itself.

.. note::
   The plugin is only available when the optional ``ephysatlas`` dependencies are installed. Without
   them the ``Channel Prediction`` menu does not appear.


Installation
------------

The models and the feature extraction come from `ibleatools
<https://github.com/int-brain-lab/ibleatools>`_, which can be installed by adding the ``ephysatlas`` extra to the GUI installation:

.. code-block:: console

   pip install -e ".[ephysatlas]"


Extracting the features
-----------------------

Both models read a table of per-channel electrophysiology features, computed from the raw AP and
LF data. These are not produced by the alignment GUI; they are computed with ``ibleatools``
beforehand.

For data on your local disk, use ``compute_features_from_file``:

.. code-block:: python

    from pathlib import Path
    from ephysatlas.feature_computation import compute_features_from_file

    # Raw AP and LF binary files for the probe
    ap_file = Path('/path/to/probe00/data.ap.bin')
    lf_file = Path('/path/to/probe00/data.lf.bin')

    # Where to write the computed features
    output_dir = Path('/path/to/probe00/features')

    compute_features_from_file(ap_file=ap_file, lf_file=lf_file, output_dir=output_dir)

This writes a parquet file of per-channel features into ``output_dir``. Repeat it for each probe
you want to run the prediction on.


Pointing the GUI at the features
--------------------------------

The recommended route is to add the features file to the session YAML as the ``features`` dataset.
You can add an extra dataset to the probe in the session YAML, for example:

.. code-block:: yaml

   path: /path/to/session_data

   probes:
     probe_00:
       datasets:
         spike_sorting:
           path: probe_00/spike_sorting
         picks:
           path: probe_00/picks
         features:
           path: probe_00/features/raw_ephys_features.pqt

The path follows the same resolution rules as the other datasets, so it can be relative to the
probe, configuration or top-level ``path``.

Alternatively a features file can be chosen at runtime from
``Plugins -> Channel Prediction -> Load features file…``.

.. note::
   A features file chosen from the menu applies to the session that is currently loaded only. It is
   cleared whenever new data is loaded, so for a session you return to it is better to add the
   ``features`` dataset to the YAML.


Loading a model
---------------

Once the features are available, load a model from the ``Plugins -> Channel Prediction`` menu:

**Load inference model**
    Select the directory holding the trained classifier. The directory must contain a ``folds``
    subdirectory (``folds/FOLD00/`` and so on).

**Load spatial model**
    A dialog with two rows, each with its own ``Browse…`` button:

    * **Model** — the directory holding the encoder checkpoint, which must contain
      ``SE_model_*.pt``
    * **Features** — the directory holding the feature tables the encoder was trained against,
      which must contain ``raw_ephys_features*.pqt``

Building the spatial encoder takes a little time, as the model and its reference bank are read in;
progress is reported in the terminal.

.. note::
   Downloading the trained models automatically, rather than pointing the GUI at a local copy, is
   coming soon. Until the models are published, both must be loaded from a local directory.

To avoid selecting the same directories every time, the GUI can be launched with the paths already
filled in. See ``examples/launch_with_local_prediction.py`` in the repository, which opens a
session and pre-populates the model paths so the dialogs are skipped.


Where the results appear
------------------------

The predictions and the features are added to three of the menu bars:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Menu
     - Added entries
   * - ``Region Plots``
     - ``Inference Model`` and ``Inference Cumulative`` for the inference model, and
       ``Spatial Encoder`` for the spatial encoder
   * - ``Feature Plots``
     - A single ``Ephys Atlas`` entry, tiling every feature side by side
   * - ``Probe Plots``
     - One entry per feature, named ``Ephys Atlas - <feature>``

Predicted regions
~~~~~~~~~~~~~~~~~

The region entries are added alongside the ``Allen``, ``Beryl`` and ``Cosmos`` mappings, so the
predicted regions can be flipped against the histology regions using the same shortcut:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Shortcut
     - Action
   * - :kbd:`Alt+5` / :kbd:`Shift+Alt+5`
     - Region plots (forward / backward)

The entries only appear once the corresponding model has been loaded.

Features
~~~~~~~~

Every feature in the table is also made available as a plot, normalised across the channels of the
shank:

- the ``Ephys Atlas`` entry in the ``Feature Plots`` menu shows all of the features at once, tiled
  side by side, for comparing them against each other
- the ``Ephys Atlas - <feature>`` entries in the ``Probe Plots`` menu show one feature at a time
  laid out on the probe geometry, in the same way as the other probe plots

As with any probe plot, the channels shown on the histology slice are coloured by the selected
feature.
