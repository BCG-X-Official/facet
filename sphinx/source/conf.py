"""
Configuration file for the Sphinx documentation builder.

For a full list of options see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import logging
import sys

from sphinx.application import Sphinx

log = logging.getLogger(name=__name__)
log.setLevel(logging.INFO)


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# noinspection DuplicatedCode
def _set_paths() -> None:
    import sys
    import os

    module_paths = ["facet", "pytools", "sklearndf"]

    if "cwd" not in globals():
        # noinspection PyGlobalUndefined
        global cwd
        cwd = os.path.join(os.getcwd(), os.pardir, os.pardir)
    print(f"working dir is '{os.getcwd()}'")
    for module_path in module_paths:
        if module_path not in sys.path:
            # noinspection PyUnboundLocalVariable
            sys.path.insert(0, os.path.abspath(f"{cwd}/{os.pardir}/{module_path}/src"))
            print(f"added `{sys.path[0]}` to python paths")


_set_paths()

log.info(f"sys.path = {sys.path}")

# -- Project information -----------------------------------------------------

project = "facet"
# noinspection PyShadowingBuiltins
copyright = "2020, The Boston Consulting Group (BCG)"
author = "FACET Team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.imgmath",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

# -- Options for autodoc / autosummary -------------------------------------------------

# generate autosummary even if no references
autosummary_generate = True

# always overwrite generated auto summaries with newly generated versions
autosummary_generate_overwrite = True

autodoc_default_options = {
    "no-ignore-module-all": True,
    "inherited-members": True,
    "imported-members": True,
    "no-show-inheritance": True,
    "member-order": "groupwise",
}

nbsphinx_allow_errors = True
nbsphinx_timeout = 60 * 15  # 15 minutes due to tutorial/model notebook

# add intersphinx mapping
intersphinx_mapping = {
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3.6", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "shap": ("https://shap.readthedocs.io/en/latest", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
}

intersphinx_collapsible_submodules = {
    "pandas.core.frame": "pandas",
    "pandas.core.series": "pandas",
    "pandas.core.panel": "pandas",
    "pandas.core.indexes.base": "pandas",
    "pandas.core.indexes.multi": "pandas",
    "mtrand": "numpy.random",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = [".rst", ".md", ".ipynb"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*/.ipynb_checkpoints/*"]

# -- Options for sphinx_autodoc_typehints ----------------------------------------------
set_type_checking_flag = False
typehints_fully_qualified = False
always_document_param_types = False

# -- Options for Math output -----------------------------------------------------------

imgmath_image_format = "svg"
imgmath_use_preview = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/Gamma_Facet_Logo_RGB_LB.svg"
latex_logo = html_logo

# Class documentation to include docstrings both global to the class, and from __init__
autoclass_content = "both"

# -- End of options section ------------------------------------------------------------


def setup(app: Sphinx) -> None:
    """
    Add event handlers to the Sphinx application object
    :param app: the Sphinx application object
    """

    from pytools.sphinx import AddInheritance, CollapseModulePaths

    AddInheritance(collapsible_submodules=intersphinx_collapsible_submodules).connect(
        app=app
    )

    CollapseModulePaths(
        collapsible_submodules=intersphinx_collapsible_submodules
    ).connect(app=app, priority=100000)

    app.add_css_file(filename="css/facet.css")
