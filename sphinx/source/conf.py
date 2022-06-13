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
from contextlib import contextmanager

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
        remote_pytools_path = f"https://raw.githubusercontent.com/BCG-Gamma/pytools/{pytools_branch}/sphinx/"
        files = [
            "base/make_base.py",
            "base/make_util.py",
            "base/conf_base.py",
            "source/_static_base/js/versions.js",
            "source/_templates/getting-started-header.rst"
        ]
        with TemporaryDirectory() as tmp_dir:
            for file in files:
                target = Path(tmp_dir, file)
                target.parent.mkdir(exist_ok=True, parents=True)
                request.urlretrieve(parse.urljoin(remote_pytools_path, file), target)
            yield os.path.join(tmp_dir, "base")

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
    sys.path.insert(
        0,
        pytools_path
    )

    from conf_base import set_config

    # ----- custom configuration -----

    set_config(
        globals(),
        project="facet",
        modules=["facet"],
        html_logo="_static/Gamma_Facet_Logo_RGB_LB.svg",
    )
