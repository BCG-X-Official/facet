#!/usr/bin/env python3
"""
Make sphinx documentation using the makefile in pytools
"""

import os
import sys
from pathlib import Path
from urllib import request, parse
from tempfile import TemporaryDirectory
from packaging import version
from contextlib import contextmanager
from zipfile import ZipFile

@contextmanager
def ensure_path(path):
    Path(path).mkdir(exist_ok=True, parents=True)
    yield path

def get_required_pytools_branch():
    """
    Get the branch name for the pytools branch

    Uses the facet version number 1.2.3 and returns 1.2.x
    """
    import facet

    facet_release = version.parse(facet.__version__)
    pytools_branch = ".".join([
        str(facet_release.major),
        str(facet_release.minor),
        "x"
    ])
    return pytools_branch

    return facet.__version__


@contextmanager
def ensure_pytools_sphinx_deps(local_pytools_path):
    if os.path.exists(local_pytools_path):
        yield local_pytools_path
    else:
        pytools_branch = get_required_pytools_branch()
        remote_pytools_path = f"https://github.com/BCG-Gamma/pytools/archive/{pytools_branch}.zip"
        with ensure_path(".cache") as tmp_dir:
            target = os.path.join(tmp_dir, "pytools.zip")
            request.urlretrieve(remote_pytools_path, target)
            ZipFile(target).extractall(tmp_dir)
            os.rename(os.path.join(tmp_dir, f"pytools-{pytools_branch}"), os.path.join(tmp_dir, f"pytools"))
            yield os.path.join(tmp_dir, "pytools", "sphinx", "base")

def make() -> None:
    """
    Run the common make file available in the pytools repo
    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(cwd)

    local_pytools_path = os.path.normpath(
        os.path.join(cwd, os.pardir, os.pardir, "pytools", "sphinx", "base")
    )

    with ensure_pytools_sphinx_deps(local_pytools_path) as pytools_path:
        sys.path.insert(
            0,
            pytools_path,
        )
        sys.path.insert(0, os.path.join(pytools_path, os.pardir, os.pardir, "src")) # pytools src
        # noinspection PyUnresolvedReferences
        from make_base import make
        from conf_base import set_config

        make(modules=["facet"])


if __name__ == "__main__":
    make()
