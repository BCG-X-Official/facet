#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Drawer and styles for dendrograms

The class :class:`DendrogramDrawer` draws dendrograms based
on :class:`~linkage.LinkageTree` and a :class:`DendrogramStyle`.
"""

import logging

# noinspection PyProtectedMember
from ._dendrogram._draw import DendrogramDrawer
# noinspection PyProtectedMember
from ._dendrogram._linkage import LinkageTree, Node
# noinspection PyProtectedMember
from ._dendrogram._style import (
    DendrogramFeatMapStyle,
    DendrogramLineStyle,
    DendrogramReportStyle,
    DendrogramStyle,
)

log = logging.getLogger(__name__)

__all__ = [
    "DendrogramDrawer",
    "DendrogramFeatMapStyle",
    "DendrogramLineStyle",
    "DendrogramReportStyle",
    "DendrogramStyle",
    "LinkageTree",
]
