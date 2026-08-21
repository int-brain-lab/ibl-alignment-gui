# Configuration file for the Sphinx documentation builder.

# For a full list of options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import date
from pathlib import Path

from ibl_alignment_gui import __version__

project_root = Path(__file__).parents[2].resolve()
sys.path.insert(0, project_root)


# -- Project information -----------------------------------------------------
project = 'IBL Alignment GUI'
copyright = f'{date.today().year}, International Brain Laboratory'  # noqa: A001
author = 'International Brain Laboratory'
release = '.'.join(__version__.split('.')[:3])
version = '.'.join(__version__.split('.')[:3])
rst_prolog = f"""
.. |version_code| replace:: ``{version}``
"""

html_context = {
    'display_github': False,
    'github_user': 'int-brain-lab',
    'github_repo': 'ibl-alignment-gui',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings.

extensions = [
    'myst_parser',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx_github_style',
    'sphinx_togglebutton',
    'sphinx_copybutton',
    'sphinx_design',
]
source_suffix = ['.rst', '.md']


# Napoleon settings (for your NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# 'head' rather than the default 'last_tag': the repository has no tags yet, and
# sphinx_github_style warns when git describe finds none.
linkcode_blob = 'head'
linkcode_link_text = ' '
pygments_style = 'default'
highlight_language = 'python3'

# -- Settings for automatic API generation -----------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

# The optional Allen/DocDB stack is not installed for the docs build; mocking it lets the
# backends.allen modules be imported and documented.
autodoc_mock_imports = [
    'aind_data_access_api',
    'aind_data_schema',
    'aind_data_schema_models',
    'aind_qcportal_schema',
    'aws_requests_auth',
    'boto3',
    'ants',
]

# Qt signals are class attributes whose docstring comes from PyQt (``pyqtSignal(*types, ...)``),
# which is not valid reStructuredText. They carry no useful documentation, so skip them.
def _skip_qt_signals(app, what, name, obj, skip, options):
    if type(obj).__name__ in ('pyqtSignal', 'pyqtBoundSignal'):
        return True
    return skip


def setup(app):
    """Register the sphinx extension hooks for this project."""
    app.connect('autodoc-skip-member', _skip_qt_signals)
