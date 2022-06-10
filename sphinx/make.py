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

    local_pytools_path = os.path.normpath(
        os.path.join(cwd, os.pardir, os.pardir, "pytools", "sphinx", "base")
    )
    if not os.path.exists(local_pytools_path):
        raise ModuleNotFoundError(f"Local pytools repo is needed to build docs, but"
                                  f"was not found in {local_pytools_path}")
    sys.path.insert(
        0,
        local_pytools_path,
    )
    # noinspection PyUnresolvedReferences
    from make_base import make

    make(modules=["pytools", "sklearndf", "facet"])


if __name__ == "__main__":
    make()
