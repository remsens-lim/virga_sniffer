# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))
sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../example/'))
# package_path = os.path.abspath('../..')
os.environ['PYTHONPATH'] = ':'.join((os.path.abspath('../example/'),
                                     os.environ.get('PYTHONPATH', '')))

# -- Project information -----------------------------------------------------

project = 'Virga-Sniffer'
copyright = '2022, Jonas Witthuhn, Johannes Röttenbacher, Heike Kalesse-Los'
author = 'Jonas Witthuhn, Johannes Röttenbacher, Heike Kalesse-Los'

# The full version, including alpha/beta/rc tags
release = 'v1.0.0'


# -- General configuration ---------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
]

extlinks = {
    "doi": ("https://doi.org/%s", "doi:%s"),
}

graphviz_output_format = "svg"

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'xarray': ('https://xarray.pydata.org/en/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']



napoleon_type_aliases = {
    # general terms
    "sequence": ":term:`sequence`",
    "iterable": ":term:`iterable`",
    "callable": ":py:func:`callable`",
    "dict_like": ":term:`dict-like <mapping>`",
    "dict-like": ":term:`dict-like <mapping>`",
    "path-like": ":term:`path-like <path-like object>`",
    "mapping": ":term:`mapping`",
    "file-like": ":term:`file-like <file-like object>`",
    # special terms
    # "same type as caller": "*same type as caller*",  # does not work, yet
    # "same type as values": "*same type as values*",  # does not work, yet
    "timedelta": "~datetime.timedelta",
    "string": ":class:`string <str>`",
    # numpy terms
    "array_like": ":term:`array_like`",
    "array_like": ":term:`array_like <array_like>`",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_theme_options = {
    "fixed_sidebar": "true",
    "logo": "lim.jpg",
    "logo_name": "true",
    "description": "University of Leipzig, Institute for Meteorology"
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- MyST Config -------------------------------------------------------------
# depth of autogenerated header anchors
myst_heading_anchors = 3