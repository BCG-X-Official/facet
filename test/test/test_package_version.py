import logging
from os import environ

from pytest import mark

from pytools.build import validate_release_version

import facet

log = logging.getLogger(__name__)


PACKAGE_VERSION = facet.__version__
MODULE_NAME = facet.__name__
PACKAGE_NAME = "gamma-facet"


@mark.skipif(
    condition=environ.get("RUN_PACKAGE_VERSION_TEST", "") != MODULE_NAME,
    reason=f"build is not for a {PACKAGE_NAME} release",
)
def test_package_version() -> None:
    validate_release_version(package=PACKAGE_NAME, version=PACKAGE_VERSION)
