Usage
=====

Basic Usage
-----------

To launch the alignment GUI with default settings, run:

.. code-block:: console

   alignment-gui

This opens the GUI window. You can then load your data by clicking the ``...`` button in the top-right corner and selecting the directory containing your data files.

.. note::
   This mode assumes that all required data (spike sorting output, raw electrophysiology recordings, probe trajectory files, and histology volumes) are located within the same directory.


Specifying Data Directories with YAML
--------------------------------------

For finer control over data locations, you can provide a YAML configuration file that explicitly specifies the paths to required inputs.

Launch the GUI using:

.. code-block:: console

   alignment-gui -y path/to/config.yaml


Required and Optional Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each YAML file must define paths for the following datasets:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Dataset
     - Description
   * - ``spike_sorting``
     - Directory containing spike sorting output (e.g., Kilosort, PyKilosort)
   * - ``picks``
     - Directory containing probe trajectory pick files

The following datasets are optional:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Dataset
     - Description
   * - ``raw_ephys``
     - Directory containing raw electrophysiology recordings
   * - ``processed_ephys``
     - Directory containing processed electrophysiology data. If not provided, defaults to ``raw_ephys`` path
   * - ``histology``
     - Directory containing histology volumes (typically defined in ``defaults``)
   * - ``output``
     - Directory where alignment outputs will be saved. If not provided, defaults to ``spike_sorting`` path


Single-Probe Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest configuration specifies paths for a single probe.

**Example YAML:**

.. code-block:: yaml

   defaults:
     histology:
       path: /path/to/histology

   probes:
     probe_00:
       datasets:
         spike_sorting:
           path: /path/to/probe_00/spike_sorting
         processed_ephys:
           path: /path/to/probe_00/processed_ephys
         raw_ephys:
           path: /path/to/probe_00/raw_ephys
         picks:
           path: /path/to/probe_00/picks
         output:
           path: /path/to/probe_00/alignment_outputs

**Launch command:**

.. code-block:: console

   alignment-gui -y path/to/single_probe.yaml


Multi-Probe Configuration
--------------------------

The GUI supports visualization of up to four probes simultaneously. This is particularly useful for multi-shank probes (e.g., Neuropixels 2.0 with 4 shanks), where data from each shank have been spike-sorted independently.

.. note::
   Probes are visualized simultaneously, but alignment is performed independently for each probe.

.. warning::
   When multiple probes are specified, the GUI does not split data automatically. Each probe's data must already be separated on disk.

**Example: four-probe configuration**

.. code-block:: yaml

   path: /path/to/session_data

   defaults:
     histology:
       path: /common/path/to/histology

   probes:

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

.. code-block:: console

   alignment-gui -y path/to/multi_probe.yaml


Dual Configuration Mode
------------------------

Dual configuration mode allows side-by-side comparison of two separate configurations within the same GUI window.

**Typical use cases:**

- Comparing different channel maps (e.g., dense vs. sparse)
- Evaluating spike sorting parameters
- Comparing preprocessing pipelines

**Key characteristics:**

- Two configurations are displayed side by side
- Alignment is performed jointly across both configurations
- Both configurations must reference the same probe identifiers

**Example: dual-configuration YAML**

.. code-block:: yaml

   defaults:
     histology:
       path: /path/to/histology

   configurations:
     dense:
       path: /path/to/dense_session
       probes:
         probe_00:
           datasets:
             spike_sorting:
               path: alf/probe00/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00
             picks:
               path: alf/probe00/kilosort
             output:
               path: alignment_outputs

     sparse:
       path: /path/to/sparse_session
       probes:
         probe_00:
           datasets:
             spike_sorting:
               path: alf/probe00/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00
             picks:
               path: alf/probe00/kilosort
             output:
               path: alignment_outputs

**Launch command:**

.. code-block:: console

   alignment-gui -y path/to/dual_config.yaml

.. note::
   Configuration keys (e.g., ``dense`` and ``sparse``) may be named arbitrarily.


Combining Dual Configuration and Multi-Probe Modes
---------------------------------------------------

Dual configuration mode can be combined with multi-probe mode to compare multiple probes across two configurations simultaneously.

.. important::
   Both configurations must contain the same number of probes with matching probe identifiers.

**Example: dual + multi-probe YAML**

.. code-block:: yaml

   defaults:
     histology:
       path: /path/to/histology

   configurations:
     dense:
       path: /path/to/dense_session
       probes:
         shank_0:
           datasets:
             spike_sorting:
               path: alf/probe00a/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00a
             picks:
               path: alf/probe00a/kilosort
             output:
               path: alignment_outputs/probe00a

         shank_1:
           datasets:
             spike_sorting:
               path: alf/probe00b/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00b
             picks:
               path: alf/probe00b/kilosort
             output:
               path: alignment_outputs/probe00b

     sparse:
       path: /path/to/sparse_session
       probes:
         shank_0:
           datasets:
             spike_sorting:
               path: alf/probe00a/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00a
             picks:
               path: alf/probe00a/kilosort
             output:
               path: alignment_outputs/probe00a

         shank_1:
           datasets:
             spike_sorting:
               path: alf/probe00b/kilosort
             raw_ephys:
               path: raw_ephys_data/probe00b
             picks:
               path: alf/probe00b/kilosort
             output:
               path: alignment_outputs/probe00b

**Launch command:**

.. code-block:: console

   alignment-gui -y path/to/dual_multi_probe.yaml


Advanced YAML Configuration
----------------------------

The YAML system supports defaults and hierarchical path resolution to minimize repetition when working with multiple probes or configurations.

Using Defaults
~~~~~~~~~~~~~~

The ``defaults`` section allows you to specify common paths and settings that apply to all probes and configurations. This is particularly useful for:

- Shared resources like histology volumes that are common across multiple sessions
- Standard directory names that follow consistent naming conventions

**Example:**

.. code-block:: yaml

   defaults:
     histology:
       path: /common/path/to/histology
     spike_sorting:
       path: kilosort
     raw_ephys:
       path: spikeglx

**How defaults work:**

- Default paths are applied when a dataset path is not explicitly specified
- They can be overridden at any level (configuration, probe, or dataset)
- An empty dataset specification (``{}``) uses the default path

.. tip::
   Use defaults to establish naming conventions across your project. For example, if you always save Kilosort output in a directory called ``kilosort``, specify this in defaults rather than repeating it for each probe.


Using Base Paths
~~~~~~~~~~~~~~~~

The YAML configuration supports ``path`` keys at multiple hierarchical levels to define base directories for resolving relative paths. This allows you to avoid repeating common path prefixes across multiple datasets.

**Path Resolution Priority**

When a relative path is specified for a dataset, the system searches for a base directory in the following order:

1. **Dataset-level path** — explicitly defined within the dataset's ``path`` field
2. **Default path** — defined in the ``defaults`` section
3. **Per-probe path** — defined within an individual probe configuration
4. **Per-configuration path** — defined within an individual configuration
5. **Top-level path** — defined at the root of the YAML file

During the search for the base directory it will join together relative paths to build the final absolute path.

If a dataset uses an empty specification (``{}``), the default path is appended to the resolved base path.

.. note::
   Absolute paths (those starting with ``/`` on Unix/Linux/macOS or a drive letter on Windows) always take precedence and are used as-is, ignoring any base paths defined at higher levels.

.. tip::
   Define base paths at the highest applicable level to minimize repetition. For example, if all probes share a common session directory, define it at the configuration level rather than repeating it for each probe.


Examples Using Base Paths
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each example below demonstrates the path resolution logic used to determine the final dataset paths.

**1. Top-level base path**

.. code-block:: yaml

   path: /mnt/data

   probes:
     probe_00:
       datasets:
         spike_sorting:
           path: probe_00/kilosort

Resolves to::

   /mnt/data/probe_00/kilosort

**2. Per-configuration base path**

.. code-block:: yaml

   path: /mnt/data

   configurations:
     dense:
       path: dense_session
       probes:
         probe_00:
           datasets:
             spike_sorting:
               path: kilosort

Resolves to::

   /mnt/data/dense_session/kilosort

**3. Per-probe base path**

.. code-block:: yaml

   path: /mnt/data

   probes:
     probe_00:
       path: probe_00
       datasets:
         spike_sorting:
           path: kilosort

Resolves to::

   /mnt/data/probe_00/kilosort

**4. Dataset-level path (highest priority)**

.. code-block:: yaml

   path: /mnt/data

   probes:
     probe_00:
       path: probe_00
       datasets:
         spike_sorting:
           path: /custom/location/kilosort

Resolves to::

   /custom/location/kilosort

**5. Using defaults with empty dataset specification**

.. code-block:: yaml

   path: /mnt/data

   defaults:
     spike_sorting:
       path: pykilosort

   probes:
     probe_00:
       path: probe_00
       datasets: {}

Resolves to::

   /mnt/data/probe_00/pykilosort


Complete Example with Path Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This comprehensive example demonstrates how different path resolution levels work together:

.. code-block:: yaml

   path: /mnt/s0/Data

   defaults:
     histology:
       path: /common/histology/subject_001
     spike_sorting:
       path: pykilosort
       backend: phylib
     raw_ephys:
       path: spikeglx
       backend: spikeglx

   configurations:
     dense:
       path: dense_session
       probes:
         probe_00:
           path: probe_00
           datasets:
             processed_ephys:
               path: pykilosort
             picks:
               path: /custom/path/to/picks  # Absolute path overrides
             output:
               path: alignment_outputs

     sparse:
       path: sparse_session
       probes:
         probe_00:
           path: probe_00
           datasets:
             spike_sorting:
               path: kilosort
             processed_ephys:
               path: raw_ephys_data
             raw_ephys: {}
             picks:
               path: kilosort
             output: {}  # Defaults to spike_sorting path

**Resolved paths for this configuration:**

.. list-table::
   :header-rows: 1
   :widths: 30 30 60

   * - Configuration / Probe
     - Dataset
     - Resolved Path

   * - **Dense / probe_00**
     - spike_sorting
     - /mnt/s0/Data/dense_session/probe_00/pykilosort

   * -
     - processed_ephys
     - /mnt/s0/Data/dense_session/probe_00/pykilosort

   * -
     - raw_ephys
     - /mnt/s0/Data/dense_session/probe_00/spikeglx

   * -
     - picks
     - /custom/path/to/picks

   * -
     - histology
     - /common/histology/subject_001

   * -
     - output
     - /mnt/s0/Data/dense_session/probe_00/alignment_outputs

   * - **Sparse / probe_00**
     - spike_sorting
     - /mnt/s0/Data/sparse_session/probe_00/kilosort

   * -
     - processed_ephys
     - /mnt/s0/Data/sparse_session/probe_00/raw_ephys_data

   * -
     - raw_ephys
     - /mnt/s0/Data/sparse_session/probe_00/spikeglx

   * -
     - picks
     - /mnt/s0/Data/sparse_session/probe_00/kilosort

   * -
     - histology
     - /common/histology/subject_001

   * -
     - output
     - /mnt/s0/Data/sparse_session/probe_00/kilosort
