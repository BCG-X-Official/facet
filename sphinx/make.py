#!/usr/bin/env python3
"""
Make sphinx documentation using the makefile in pytools
"""

import os
import sys

if __name__ == "__main__":

    cwd = os.path.dirname(__file__)
    os.chdir(cwd)

    sys.path.append(os.path.realpath(os.path.join(cwd, os.pardir, os.pardir)))
    # noinspection PyUnresolvedReferences
    from pytools.sphinx.make import run_make

    run_make()
