# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# ensure, yieldengine modules are found:
import os
import sys

sys.path.insert(0, os.path.abspath("./../../src"))

# -- Project information -----------------------------------------------------

project = "yield-engine"
copyright = "2019, Joerg Schneider"
author = "Joerg Schneider"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

nbsphinx_allow_errors = True

# add intersphinx mapping
intersphinx_mapping = {
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
    #  'pd': ('http://pandas.pydata.org/pandas-docs/dev', None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/3.6", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": (
        "https://scikit-learn.org/stable",
        (None, "./_intersphinx/sklearn-objects.inv"),
    ),
    "shap": ("https://shap.readthedocs.io/en/latest/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autoclass_content = "class"
