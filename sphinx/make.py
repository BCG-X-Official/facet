#!/usr/bin/env python3
"""
Make sphinx documentation using the pytools make utility
"""
import os
from urllib import request

BRANCH = "2.0.x"


if __name__ == "__main__":

    # noinspection PyUnusedLocal
    def run_make(branch: str, working_directory: str) -> None:
        """Stub, overwritten by bootstrap.py"""

    # run the common make file available in the pytools repo
    with request.urlopen(
        f"https://raw.githubusercontent.com/BCG-Gamma/pytools/{BRANCH}"
        f"/sphinx/base/bootstrap.py"
    ) as response:
        exec(response.read().decode("utf-8"), globals())

    run_make(branch=BRANCH, working_directory=os.path.dirname(__file__))
