#!/usr/bin/env python3
"""
Make sphinx documentation using the makefile in pytools
"""

import os
import sys


def run_make() -> None:
    """
    Run the common make file available in the pytools repo
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    sys.path.append(os.path.normpath(os.path.join(cwd, os.pardir, os.pardir)))
    # noinspection PyUnresolvedReferences
    from pytools.sphinx.make import run_make

    run_make()


if __name__ == "__main__":
    run_make()
