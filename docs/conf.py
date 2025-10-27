# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'circleclust'
copyright = '2025, Tim Pyrkov'
author = 'Tim Pyrkov'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'myst_parser',          # Markdown support (README.md)
    'nbsphinx',             # Jupyter notebooks
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

autosummary_generate = True

# Autodoc configuration for detailed function documentation
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'inherited-members': True,
}

# Enable detailed function signatures
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True

# On RTD, the package will be installed; for safety we can mock heavy libs
autodoc_mock_imports = [
    'matplotlib',
    'numpy',
    'pandas',
    'PIL',
    'requests',
    'scipy',

]

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'tasklist',
]
myst_heading_anchors = 3

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = 'img/caption.png'
pygments_style = 'sphinx'
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']



# Notebook parameters
nbsphinx_allow_errors = True

master_doc = 'index'
