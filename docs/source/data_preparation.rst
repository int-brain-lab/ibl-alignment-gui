Data Preparation
================

This section describes how to prepare your data for use with the IBL Alignment GUI.

The GUI requires:

- Spike-sorted electrophysiology data in the phylib format
- Extracted raw electrophysiology features
- Probe trajectory coordinates in the brain atlas


Preparing Electrophysiology Data
---------------------------------

The IBL Alignment GUI requires spike-sorted data in the phylib format, along with extracted raw electrophysiology features computed from the AP and LFP data.

Using SpikeGLX and Kilosort
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you recorded data using SpikeGLX and spike-sorted using Kilosort or pykilosort, use the following code to convert your data and extract the necessary features:

.. code-block:: python

    from pathlib import Path
    from ibl_alignment_gui.convertors import extract_ephys

    # Path to Kilosort output
    ks_path = Path('/path/to/kilosort/output')

    # Path to raw ephys data
    ephys_path = Path('/path/to/raw/ephys/data')

    # Output path
    out_path = Path('/path/to/output')

    extract_ephys(ks_path, ephys_path, out_path)

.. warning::
    Ensure the output path is **not** the same as the Kilosort path to avoid overwriting existing files.

Using Other Recording or Spike Sorting Software
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you recorded data using OpenEphys or spike-sorted using other software, we recommend using SpikeInterface to convert your data to the phylib format.

SpikeInterface provides a specific converter to export a `SpikeSortingAnalyzer` to the format required by the IBL Alignment GUI:

- `SpikeInterface documentation <https://spikeinterface.readthedocs.io/en/latest/>`_
- `IBL exporter module <https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/exporters/to_ibl.py>`_


Preparing Trajectory Data
--------------------------

The IBL Alignment GUI requires the location of the probe trajectory in the brain atlas. This is typically obtained via probe track reconstruction from histology images.

Available Tools
~~~~~~~~~~~~~~~

There are several tools available for probe track reconstruction:

- BrainRegister
- Lasagna
- SHARP-Track
- Herbs

Probe Tracing Using brainreg-segment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`brainreg` and `brainreg-segment` are tools for registering histology image stacks and tracing probe locations in the brain.

**Tutorial**: See the `brainreg-segment tutorial <https://brainglobe.info/tutorials/segmenting-1d-tracks.html>`_ for detailed instructions.

**Important notes:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Aspect
     - Details
   * - **Tracing space**
     - Must be done in registered atlas space (not original sample space)
   * - **Spline points**
     - Reduce to < 100 when fitting the track
   * - **Export**
     - Click **Export to brainrender** to output the `.npy` coordinate file

**Converting brainreg output to GUI format:**

.. code-block:: python

    import numpy as np
    from pathlib import Path
    import json
    from iblatlas.atlas import AllenAtlas

    atlas = AllenAtlas(25)

    # Path to brainreg track output
    brainreg_path = Path('/path/to/brainreg/output/tracks/track_1.npy')

    # Load coordinates in CCF space (order: apdvml, origin: top-left-front voxel)
    xyz_apdvml = np.load(brainreg_path)

    # Convert to IBL space (order: mlapdv, origin: bregma)
    xyz_mlapdv = atlas.ccf2xyz(xyz_apdvml, ccf_order='apdvml') * 1e6

    xyz_picks = {'xyz_picks': xyz_mlapdv.tolist()}

    # Save to output directory
    output_path = Path('/path/to/output')
    with open(output_path / 'xyz_picks.json', 'w') as f:
        json.dump(xyz_picks, f, indent=2)


Probe Tracing Using Lasagna
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lasagna is another tool for tracing probe tracks in histology images registered to the Allen atlas.

**Tutorial**: See the `Lasagna probe tracing guide <https://github.com/SainsburyWellcomeCentre/lasagna/wiki/Use-case:-tracing-electrode-tracks>`_ for instructions.

**Important notes:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Aspect
     - Details
   * - **Image transformations**
     - Do not apply rotations, flips, or mirrors when tracing
   * - **Tracing space**
     - Trace in histology already registered to Allen atlas, or apply registration transform if tracing in original space
   * - **Output line**
     - Save the `_pts` line, not the `_fit` line

**Converting Lasagna output to GUI format:**

.. code-block:: python

    from ibllib.pipes.histology import load_track_csv
    from pathlib import Path
    import json

    # Path to Lasagna tracing output
    file_track = '/path/to/lasagna/tracing_pts.csv'

    # Load and convert coordinates
    xyz = load_track_csv(file_track) * 1e6
    xyz_picks = {'xyz_picks': xyz.tolist()}

    # Save to output directory
    output_path = Path('/path/to/output')
    with open(output_path / 'xyz_picks.json', 'w') as f:
        json.dump(xyz_picks, f, indent=2)


Coordinate Systems
------------------

The IBL Alignment GUI uses coordinates relative to bregma with the following convention:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Axis
     - Description
   * - **x**
     - Medial-lateral (ML)
   * - **y**
     - Anterior-posterior (AP)
   * - **z**
     - Dorsal-ventral (DV)

Bregma is defined at:

- ML = 5739 μm
- AP = 5400 μm
- DV = 332 μm

from the front-top-left corner (from the mouse's point of view) of the Allen CCF data volume.


Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have probe tracks in the Allen CCF coordinate framework, use the following code to transform between coordinate systems:

**CCF to Bregma (mlapdv order):**

.. code-block:: python

    from iblatlas.atlas import AllenAtlas
    import numpy as np

    # Initialize atlas (25 μm resolution)
    brain_atlas = AllenAtlas(25)

    # Example coordinates in μm with CCF origin
    ccf_mlapdv = np.array([[3000, 4000, 3000], [6000, 6000, 500]], dtype=float)

    # Transform to Bregma origin
    bregma_mlapdv = brain_atlas.ccf2xyz(ccf_mlapdv, ccf_order='mlapdv')

**CCF to Bregma (apdvml order):**

.. code-block:: python

    # Example coordinates in μm with CCF origin (apdvml order)
    ccf_apdvml = np.array([[3000, 4000, 3000], [6000, 6000, 500]], dtype=float)

    # Transform to Bregma origin (output in mlapdv order)
    bregma_mlapdv = brain_atlas.ccf2xyz(ccf_apdvml, ccf_order='apdvml')

**Bregma to CCF (mlapdv order):**

.. code-block:: python

    # Example coordinates in m with Bregma origin
    bregma_mlapdv = np.array([[2000, 4000, 0], [4000, -1000, -4000]]) / 1e6

    # Transform to CCF origin
    ccf_mlapdv = brain_atlas.xyz2ccf(bregma_mlapdv, ccf_order='mlapdv')

**Bregma to CCF (apdvml order):**

.. code-block:: python

    # Example coordinates in m with Bregma origin
    bregma_mlapdv = np.array([[2000, 4000, 0], [4000, -1000, -4000]]) / 1e6

    # Transform to CCF origin (apdvml order)
    ccf_apdvml = brain_atlas.xyz2ccf(bregma_mlapdv, ccf_order='apdvml')
