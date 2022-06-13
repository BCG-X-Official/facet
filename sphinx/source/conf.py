"""
Configuration file for the Sphinx documentation builder.

Receives majority of configuration from pytools conf_base.py
"""

import os
import sys


local_pytools_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    "pytools",
    "sphinx",
    "base",
)
sys.path.insert(0, local_pytools_path)
from conf_base import set_config

# ----- custom configuration -----

set_config(
    globals(),
    project="facet",
    modules=["facet"],
    html_logo="_static/Gamma_Facet_Logo_RGB_LB.svg",
)
