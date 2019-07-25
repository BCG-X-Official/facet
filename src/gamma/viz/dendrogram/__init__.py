#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, even not in modified form.
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

from gamma.viz.dendrogram._draw import DendrogramDrawer
from gamma.viz.dendrogram._linkage import LinkageTree, Node
from gamma.viz.dendrogram._style import (
    DendrogramStyle,
    DendrogramTextStyle,
    FeatMapStyle,
    LineStyle,
)

log = logging.getLogger(__name__)

__all__ = [
    "DendrogramDrawer",
    "DendrogramStyle",
    "FeatMapStyle",
    "LineStyle",
    "LinkageTree",
    "DendrogramTextStyle",
]
