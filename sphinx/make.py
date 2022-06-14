#!/usr/bin/env python3
"""
Make sphinx documentation using the pytools make utility
"""
from typing import Callable
from urllib import request

BRANCH = "2.0.x"


if __name__ == "__main__":
    # run the common make file available in the pytools repo
    with request.urlopen(
        f"https://raw.githubusercontent.com/BCG-Gamma/pytools/{BRANCH}"
        f"/sphinx/base/bootstrap.py"
    ) as response:
        run_make: Callable = lambda branch: ...
        exec(response.read().decode("utf-8"), globals())

    run_make(BRANCH)
