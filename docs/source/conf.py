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
    'sphinx_reredirects',
]
source_suffix = ['.rst', '.md']

# -- MyST (Markdown) settings ------------------------------------------------
# colon_fence lets sphinx-design cards, tabs and dropdowns be written as ::: blocks.
myst_enable_extensions = [
    'attrs_inline',
    'colon_fence',
    'deflist',
    'fieldlist',
    'substitution',
]
myst_heading_anchors = 3
# Markdown counterpart of |version_code| in rst_prolog, used as {{ version_code }}.
myst_substitutions = {
    'version_code': f'`{version}`',
}

# -- Redirects ---------------------------------------------------------------
# Old page -> new page, so links to the previous layout keep working.
redirects = {
    'getting_started': 'tutorials/01_first_alignment.html',
    'usage': 'reference/session_yaml.html',
    'multi_shank': 'how-to/configure/index.html',
    'data_preparation': 'how-to/prepare-data/index.html',
    'datasets': 'reference/input_datasets.html',
    'plugins': 'reference/plugins.html',
    'shortcuts': 'reference/shortcuts.html',
    'channel_prediction': 'how-to/channel-prediction/index.html',
    'developer_guide': 'contributing/developer_guide.html',
}


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
html_css_files = ['custom.css']
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

# The optional extras are not installed for the docs build; mocking them lets the modules that
# import them be imported and documented, and lets the type hints that name them resolve.
autodoc_mock_imports = [
    # allen extra: still imported by shared modules, although Allen mode is not documented
    'aind_data_access_api',
    'aind_data_schema',
    'aind_data_schema_models',
    'aind_qcportal_schema',
    'aws_requests_auth',
    'boto3',
    'ants',
    # ibl extra
    'brainbox',
    'ibllib',
    'phylib',
    # ephysatlas extra
    'ephysatlas',
]

# Allen/Code Ocean mode is not documented, so its classes and functions in shared modules are
# left out of the API reference. Its own modules are excluded in custom-module-template.rst.
_ALLEN_MEMBERS = {
    'AlignmentLoaderDocDB',
    'AlignmentUploaderDocDB',
    'AnatomicalSliceLoader',
    'ImageSpacePaths',
    'ProbeHandlerAllenYaml',
    'build_anatomical_atlas',
    'has_allen',
    'launch_app_allen',
}


# Qt signals are class attributes whose docstring comes from PyQt (``pyqtSignal(*types, ...)``),
# which is not valid reStructuredText. They carry no useful documentation, so skip them.
# Both checks share one handler: autodoc uses the first handler that returns a value, so a
# second handler connected to the same event would never be called.
def _skip_member(app, what, name, obj, skip, options):
    if type(obj).__name__ in ('pyqtSignal', 'pyqtBoundSignal'):
        return True
    if name in _ALLEN_MEMBERS:
        return True
    return skip


def setup(app):
    """Register the sphinx extension hooks for this project."""
    app.connect('autodoc-skip-member', _skip_member)
