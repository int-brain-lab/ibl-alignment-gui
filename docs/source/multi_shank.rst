Multi-Shank Data
================

The IBL Alignment GUI supports probes with multiple shanks. The GUI supports two methods for organizing multi-shank probe data.

Method 1: All Shanks in a Single Directory
-------------------------------------------

In this format, spike sorting is run on the full probe, and the output contains results from all shanks together. The extracted raw features also include data from all shanks.

The GUI automatically detects the number of shanks from the channel metadata and displays them accordingly.

**Requirements**

You must provide one probe tracing (xyz picks) file per shank in the picks directory.

For example, for a four-shank probe, the picks directory should contain:

.. code-block:: text

    xyz_picks_shank_1.json
    xyz_picks_shank_2.json
    xyz_picks_shank_3.json
    xyz_picks_shank_4.json

Each file corresponds to the trajectory of a single shank.

**Output**

The output results will be saved in the output directory with a suffix indicating the shank number:

.. code-block:: text

    channel_locations_shank_1.json
    channel_locations_shank_2.json
    channel_locations_shank_3.json
    channel_locations_shank_4.json


Method 2: Each Shank in a Separate Directory
---------------------------------------------

In this format, spike sorting is run independently for each shank, and the extracted raw features contain data from only that shank.

To load this type of data, use the **multi-probe configuration** described in the Usage section. Each shank should be treated as a separate probe and specified as its own entry in the YAML configuration file.

This allows all shanks to be displayed simultaneously within the same GUI window, while still handling them as independent probes internally.

**Example YAML configuration:**

.. code-block:: yaml

   probes:
     path: /path/to/session_data
     shank_0:
       datasets:
         spike_sorting:
           path: shank_0/spike_sorting
         processed_ephys:
           path: shank_0/processed_ephys
         raw_ephys:
           path: shank_0/raw_ephys
         picks:
           path: shank_0/picks
         output:
           path: shank_0/output

     shank_1:
       datasets:
         spike_sorting:
           path: shank_1/spike_sorting
         processed_ephys:
           path: shank_1/processed_ephys
         raw_ephys:
           path: shank_1/raw_ephys
         picks:
           path: shank_1/picks
         output:
           path: shank_1/output

      shank_2:
       datasets:
         spike_sorting:
           path: shank_2/spike_sorting
         processed_ephys:
           path: shank_2/processed_ephys
         raw_ephys:
           path: shank_2/raw_ephys
         picks:
           path: shank_2/picks
         output:
           path: shank_2/output

     shank_3:
       datasets:
         spike_sorting:
           path: shank_3/spike_sorting
         processed_ephys:
           path: shank_3/processed_ephys
         raw_ephys:
           path: shank_3/raw_ephys
         picks:
           path: shank_3/picks
         output:
           path: shank_3/output


**Launch command:**

.. code-block:: bash

    alignment-gui -y path/to/multi_shank_config.yaml

See :doc:`usage` for more multi-probe configuration examples.


Working with Multi-Shank Data
------------------------------

Regardless of which method you use, the following features are available when working with multi-shank data:

Display Modes
~~~~~~~~~~~~~

Two display modes are available:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Display Mode
     - Description
   * - **Tabbed view**
     - Each shank displayed under its own tab
   * - **Non-tabbed view**
     - All shanks displayed simultaneously

Toggle between display modes:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Shortcut
     - Action
   * - Click **Tabbed View** in Display menu
     - Toggle view mode
   * - :kbd:`Shift+T`
     - Toggle view mode


Navigation
~~~~~~~~~~

Navigate between shanks:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Shortcut
     - Action
   * - Click shank tabs
     - Switch between shanks
   * - :kbd:`Left` / :kbd:`Right`
     - Switch between shanks

The active shank is highlighted in red in the title bar of each figure.


Alignment
~~~~~~~~~

- Alignment is always performed independently per shank, regardless of how the data are organized on disk
- The Fit figure shows fit lines for all shanks simultaneously
- Up to four shanks can be visualized at once
