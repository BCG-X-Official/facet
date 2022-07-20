#!/usr/bin/env python3
"""
Make sphinx documentation using the pytools make utility
"""
import os
from typing import Callable
from urllib import request

BRANCH = "2.0.x"


if __name__ == "__main__":

    # make sure that the current working directory is where this script lives
    wd_expected = os.path.realpath(os.path.dirname(__file__))
    if os.path.realpath(os.getcwd()) != wd_expected:
        raise RuntimeError(
            "Before running this script, the working directory must be changed to "
            f"{wd_expected}"
        )

    # run the common make file available in the pytools repo
    with request.urlopen(
        f"https://raw.githubusercontent.com/BCG-Gamma/pytools/{BRANCH}"
        f"/sphinx/base/bootstrap.py"
    ) as response:
        run_make: Callable[[str], None] = lambda branch: None
        exec(response.read().decode("utf-8"), globals())

    run_make(BRANCH)
