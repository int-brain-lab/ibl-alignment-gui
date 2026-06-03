Installation
============

Prerequisites
-------------

To use the IBL Alignment GUI, you need Python 3.10 or higher installed on your system.

We recommend installing the package into a new virtual environment to avoid dependency conflicts. You can use any of the following tools:

- **uv** (recommended)
- **venv** (included with Python)
- **conda** (if you have Anaconda or Miniconda installed)


Installation Methods
--------------------

Choose one of the following installation methods based on your preferred virtual environment tool.

.. dropdown:: **Using uv (Recommended)**
   :open:

    `uv` is a fast Python package installer and resolver.

    **Install uv**

    Follow the installation instructions at: https://docs.astral.sh/uv/getting-started/installation/

    **Create and activate virtual environment**

    Open a terminal and create a new virtual environment:

    .. code-block:: console

       uv venv ibl-alignment --python 3.13

    Activate the virtual environment:

    .. tab-set::

       .. tab-item:: Linux and macOS

          .. code-block:: console

             source ibl-alignment/bin/activate

       .. tab-item:: Windows

          .. code-block:: console

             .\ibl-alignment\Scripts\activate

    **Install the package**

    Clone and install the IBL Alignment GUI:

    .. code-block:: console

       git clone https://github.com/int-brain-lab/ibl-alignment-gui.git
       cd ibl-alignment-gui
       uv pip install -e .


.. dropdown:: **Using venv**

    Python's built-in `venv` module can be used to create virtual environments.

    **Create and activate virtual environment**

    Open a terminal and create a new virtual environment:

    .. code-block:: console

       python -m venv ibl-alignment

    Activate the virtual environment:

    .. tab-set::

       .. tab-item:: Linux and macOS

          .. code-block:: console

             source ibl-alignment/bin/activate

       .. tab-item:: Windows

          .. code-block:: console

             .\ibl-alignment\Scripts\activate.bat

    **Install the package**

    Clone and install the IBL Alignment GUI:

    .. code-block:: console

       git clone https://github.com/int-brain-lab/ibl-alignment-gui.git
       cd ibl-alignment-gui
       pip install -e .

.. dropdown:: **Using conda**

    If you have Anaconda or Miniconda installed, you can use conda to create a virtual environment.

    **Create and activate virtual environment**

    Open a terminal and create a new virtual environment:

    .. code-block:: console

       conda create -n ibl-alignment python=3.13

    Activate the virtual environment:

    .. code-block:: console

       conda activate ibl-alignment

    **Install the package**

    Clone and install the IBL Alignment GUI:

    .. code-block:: console

       git clone https://github.com/int-brain-lab/ibl-alignment-gui.git
       cd ibl-alignment-gui
       pip install -e .


Verifying Installation
----------------------

To verify that the installation was successful, run:

.. code-block:: console

   python -c "import ibl_alignment_gui; print(ibl_alignment_gui.__version__)"

This should print the version number of the installed package without any errors.

If the installation was successful, you can proceed to the :doc:`getting_started` guide.
