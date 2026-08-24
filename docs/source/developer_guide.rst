Developer Guide
===============


Setting Up the Development Environment
--------------------------------------

This project uses `UV <https://github.com/astral-sh/uv>`_ as its package manager for
managing dependencies and ensuring consistent and reproducible environments. To install
UV:

.. tab-set::

   .. tab-item:: Linux and macOS

      .. code-block:: console

          curl -LsSf https://astral.sh/uv/install.sh | sh

   .. tab-item:: Windows

      .. code-block:: pwsh-session

         PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

See `UV's documentation <https://docs.astral.sh/uv/>`_ for details.

Once UV is installed, synchronize your environment with the dependencies specified in
the ``pyproject.toml`` file, including development dependencies:

.. code-block:: console

    uv sync


.. _unit_tests:

Testing and Code Quality
------------------------

Two tools are used to check the code:

* `pytest <https://docs.pytest.org/>`_ — unit-tests (located in the ``tests`` directory)
* `ruff <https://docs.astral.sh/ruff/>`_ — linting and formatting checks

Run them against your current environment with:

.. code-block:: console

    uv run pytest          # run unit-tests
    uv run ruff check      # check for linting issues
    uv run ruff format     # auto-format code

Adding ``--fix`` to ``ruff check`` will automatically correct fixable issues. Both should be
clean before a pull request is opened.

To measure how much of the code is covered by the unit-tests, run the tests with coverage
enabled and then report on the result:

.. code-block:: console

    uv run pytest --cov
    uv run coverage report

For a more detailed representation, generate an HTML report:

.. code-block:: console

    uv run coverage html

You'll find the HTML report in the folder ``htmlcov``, where you can open ``index.html``
in a web browser to view detailed coverage statistics.

.. note::
   `mypy <https://mypy-lang.org/>`_ is available in the ``typing`` dependency group and can be
   run with ``uv run mypy``, but the code base is not yet free of type errors, so it is not part
   of the checks above.


Pull Requests
-------------

All development work should be based on the ``develop`` branch. To contribute:

1. Create a new branch from ``develop``:

   .. code-block:: console

       git checkout develop
       git checkout -b your-branch-name

2. Make your changes, keeping each pull request focused on a single topic (feature,
   bugfix, refactor, etc.).

3. Before opening a pull request, ensure that all tests pass and the code is properly
   formatted (see :ref:`unit_tests`)

4. Open your pull request against the ``develop`` branch. The ``main`` branch only
   receives merges from ``develop`` as part of the release process.


Building the Documentation
--------------------------

We use `Sphinx <https://www.sphinx-doc.org/>`_ to build our documentation and
API reference. To build the documentation, run the following command:

.. code-block:: console

    uv run sphinx-build docs/source docs/build

After running this command, you can view the generated documentation in your
web browser by opening ``docs/build/index.html``.


Building the Package
--------------------

To build ibl-alignment-gui as a distributable Python package, execute the following command:

.. code-block:: console

    uv build

This command will create a distributable package of ibl-alignment-gui, in the form of a source
distribution (sdist) and a wheel (bdist_wheel). The generated package files will be
located in the ``dist`` directory.


Versioning Scheme
-----------------

ibl-alignment-gui uses `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. Its version
string (currently |version_code|) is a combination of three fields, separated by dots:

.. centered:: ``MAJOR`` . ``MINOR`` . ``PATCH``

* The ``MAJOR`` field is only incremented for breaking changes, i.e., changes that are
  not backward compatible with previous changes.
* The ``MINOR`` field will be incremented upon adding new, backward compatible features.
* The ``PATCH`` field will be incremented with each new, backward compatible bugfix
  release that does not implement a new feature.
* Optionally appended letters can be used to indicate an alpha release (``a``), a beta
  release (``b``) or a release candidate (``rc``).

On the developer side, these fields are controlled by both

   1. adjusting the variable ``version`` field in ``pyproject.toml``, and
   2. adding the corresponding version string to a commit as a
      `git tag <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`_, for instance:

      .. code-block:: console

          git tag 1.2.3
          git push origin --tags
