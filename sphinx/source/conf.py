"""
Configuration file for the Sphinx documentation builder.

Receives majority of configuration from pytools conf_base.py
"""

import os
import sys

from contextlib import contextmanager
from pathlib import Path
from urllib import request, parse
from tempfile import TemporaryDirectory
from packaging import version
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
       yield ensure_path(".cache")
       # pytools_branch = get_required_pytools_branch()
       #  remote_pytools_path = f"https://github.com/BCG-Gamma/pytools/archive/{pytools_branch}.zip"
       #  with ".cache" as tmp_dir:
       #      target = os.path.join(tmp_dir, "pytools.zip")
       #      request.urlretrieve(remote_pytools_path, target)
       #      ZipFile(target).extractall(tmp_dir)
       #      os.rename(os.path.join(tmp_dir, f"pytools-{pytools_branch}"), os.path.join(tmp_dir, f"pytools"))
       #      yield os.path.join(tmp_dir, "pytools", "sphinx", "base")

local_pytools_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    "pytools",
    "sphinx",
    "base",
)
with ensure_pytools_sphinx_deps(local_pytools_path) as pytools_path:
    sys.path.insert(0, pytools_path)
    sys.path.insert(0, os.path.join(pytools_path, os.pardir, os.pardir, "src")) # pytools src
    breakpoint()
    from conf_base import set_config

    # ----- custom configuration -----

    set_config(
        globals(),
        project="facet",
        modules=["facet"],
        html_logo="_static/Gamma_Facet_Logo_RGB_LB.svg",
    )
