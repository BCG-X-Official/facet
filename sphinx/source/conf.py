"""
Configuration file for the Sphinx documentation builder.

Receives the majority of the configuration from pytools conf_base.py
"""

import os
import sys

_dir_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "base")
sys.path.insert(0, _dir_base)

from conf_base import set_config

# ----- set custom configuration -----

set_config(
    globals(),
    project="facet",
    html_logo=os.path.join("_images", "Gamma_Facet_Logo_RGB_LB.svg"),
    intersphinx_mapping={
        "pytools": ("https://bcg-gamma.github.io/pytools/", None),
        "shap": ("https://shap.readthedocs.io/en/stable", None),
        "sklearn": ("https://scikit-learn.org/stable", None),
        "sklearndf": ("https://bcg-gamma.github.io/sklearndf/", None),
    },
)
