#!/usr/bin/env python3
"""
Make sphinx documentation using the makefile in pytools
"""

import os
import sys


def make() -> None:
    """
    Run the common make file available in the pytools repo
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    sys.path.insert(
        0,
        os.path.normpath(
            os.path.join(cwd, os.pardir, os.pardir, "pytools", "sphinx", "base")
        ),
    )
    # noinspection PyUnresolvedReferences
    from make_base import make

    make(modules=["pytools", "sklearndf", "facet"])


if __name__ == "__main__":
    make()
