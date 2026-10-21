Channel Prediction
==================

The Channel Prediction plugin predicts the brain region of each recording channel from its
electrophysiology features, so that the prediction can be compared against the histology while
aligning. It also makes the features themselves available as plots, so they can be inspected
alongside the other electrophysiology data.

Three independent models are provided:

**Region decoders (XGBoost and ProbeTransformer)**
    Region classifiers that predict a region for each channel directly from its features: an
    XGBoost classifier working channel by channel, and a ProbeTransformer that attends across all
    channels of the probe, using each channel's depth.

**Auto-alignment**
    The Ephys Atlas channel-level model predicts the features expected along the probe's histology
    track; the recording is warped onto that prediction, giving an initial alignment of the probe
    along the track, a region for each channel and the model's confidence in each channel. It only
    needs the recording's features and the histology track, so it can be used on a new recording.

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

For an IBL insertion loaded from Alyx, the models otherwise use the ephys-atlas feature table of
the vintage they were trained on (e.g. ``2026_W26``), which may differ from the latest table shown
in the feature plots. A file chosen from the menu always takes precedence. A local file must hold
the features of the model's vintage; missing columns are reported when the model runs.

.. note::
   A features file chosen from the menu applies to the session that is currently loaded only. It is
   cleared whenever new data is loaded, so for a session you return to it is better to add the
   ``features`` dataset to the YAML.


Loading a model
---------------

Once the features are available, load a model from the ``Plugins -> Channel Prediction`` menu:

**Load XGBoost region prediction model (Hugging Face)…** / **Load ProbeTransformer region prediction model (Hugging Face)…**
    The released channel-level region decoders, loaded with ``ephysatlas.load_pretrained`` and run
    with their ``predict``: a Hugging Face repository
    (``int-brain-lab/ea-decoder-channel-xgboost`` or ``int-brain-lab/ea-decoder-channel-transformer``
    by default) and release tag (``2026_W39`` by default), or a local release directory containing
    its ``ephysatlas_model.json``. Each is downloaded once and cached. The ProbeTransformer requires
    ``torch``; its menu entry is only shown when ``torch`` is installed.

**Load alignment model (Hugging Face)…**
    The released channel-level model, loaded with ``ephysatlas.load_pretrained``: a Hugging Face
    repository (``int-brain-lab/ea-encoder-channel`` by default) and release tag, or a local
    release directory. It is downloaded once and cached; it ships its molecular-context volumes
    and training neighbour bank, so nothing else is needed. Requires ``torch``.

To avoid selecting the same model every time, the GUI can be launched with it already filled in.
See ``examples/launch_with_local_prediction.py`` in the repository, which opens a session and
pre-populates the model paths so the dialogs are skipped.


Auto-alignment
--------------

Selecting ``Region Plots -> Auto-alignment`` or ``Region Plots -> Auto-alignment (confidence)``
(loading the model first if needed) aligns each shank behind a progress dialog and opens its result
window once, with:

- the histology track (top to bottom of the brain), the planned trajectory when the features table
  has one, and the inferred alignment, on a coronal and a sagittal slice
- the cost matrix of every recorded channel against every track sample, with the alignment path
- stripes along the histology track: its atlas regions, then the confidence and the recorded
  features where the probe was aligned, next to the features the model predicts along the whole
  track

The two region entries show the same alignment:

- ``Auto-alignment``: the atlas regions along the histology track, placed at the probe depths the
  alignment found
- ``Auto-alignment (confidence)``: the alignment confidence along the probe (green high, red low;
  grey beyond the probe and for channels that could not be aligned)

Each shank is aligned once and both are kept, so switching between them, or selecting one again,
does not re-align and does not reopen the result windows. To see the windows again, use
``Plugins -> Channel Prediction -> Show auto-alignment results``, which is greyed out until a shank
has been aligned. Loading a new model or features file closes the windows and starts over, and
loading a new session closes them.


Where the results appear
------------------------

The predictions and the features are added to three of the menu bars:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Menu
     - Added entries
   * - ``Region Plots``
     - ``XGBoost region prediction`` and ``ProbeTransformer region prediction`` (the most probable region per channel), each with a
       ``(cumulative)`` entry showing the stacked class probabilities, and ``Auto-alignment`` and
       ``Auto-alignment (confidence)`` for the auto-alignment
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
