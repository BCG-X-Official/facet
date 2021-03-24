"""
This package has moved to :mod:`facet.data.partition` and is mirrored here for backward
compatibility.

Please update your code to point to the new package location.
This mirror package will be removed in FACET v1.2.
"""

from pytools.api import deprecation_warning as __deprecation_warning

# noinspection PyUnresolvedReferences
from ...data.partition import *

__deprecation_warning(
    "Package facet.simulation.partition has moved to facet.data.partition, "
    "please update your import statements. "
    "Package facet.simulation.partition will be removed in FACET v1.2.",
    stacklevel=2,
)

del __deprecation_warning
