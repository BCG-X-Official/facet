#!/usr/bin/env python3
"""
Make sphinx documentation using the makefile in pytools
"""

import os
import sys
import subprocess
from pathlib import Path
from urllib import request, parse
from packaging import version

def get_required_pytools_branch():
    """
    Get the branch name for the pytools branch

    Uses the facet version number 1.2.3 and returns 1.2.x
    """
    import facet

    # TODO: remove once pytools PR330 merged
    #  https://github.com/BCG-Gamma/pytools/pull/330
    return "doc/local_sphinx_build"

    facet_release = version.parse(facet.__version__)
    pytools_branch = ".".join([
        str(facet_release.major),
        str(facet_release.minor),
        "x"
    ])
    return pytools_branch

    return facet.__version__

def make() -> None:
    """
    Run the common make file available in the pytools repo
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    pytools_branch = get_required_pytools_branch()
    target_url = f"https://raw.githubusercontent.com/BCG-Gamma/pytools/{pytools_branch}/sphinx/base/bootstrap.py"
    target = Path("base/bootstrap.py")
    target.parent.mkdir(exist_ok=True, parents=True)
    request.urlretrieve(target_url, target)
    sys.path.insert(0, str(target.parent.resolve()))
    from bootstrap import run_make
    run_make(pytools_branch)


if __name__ == "__main__":
    make()
