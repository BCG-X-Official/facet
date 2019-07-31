"""
Drawer and styles for dendrograms

The class :class:`DendrogramDrawer` draws dendrograms based
on :class:`~linkage.LinkageTree` and a :class:`DendrogramStyle`.
"""
import logging

from gamma.viz.dendrogram._draw import DendrogramDrawer
from gamma.viz.dendrogram._linkage import LinkageTree, Node
from gamma.viz.dendrogram._style import (
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
