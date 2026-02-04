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
    'github_version': 'master',
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

linkcode_link_text = ' '
pygments_style = 'default'
highlight_language = 'python3'

# -- Settings for automatic API generation -----------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
