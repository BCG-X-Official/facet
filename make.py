#!/usr/bin/env python3
"""
call the Python make file for the common conda build process residing in 'pytools'
"""

import os
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PYTOOLS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, "pytools"))
sys.path.insert(0, PYTOOLS_DIR)

# noinspection PyUnresolvedReferences
from make import run_make

run_make()
