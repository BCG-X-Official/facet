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

import logging
from typing import *

import numpy as np

from gamma.viz import ChartDrawer
from gamma.viz.dendrogram._linkage import LinkageTree, Node
from gamma.viz.dendrogram._style import DendrogramStyle

log = logging.getLogger(__name__)


class _SubtreeInfo(NamedTuple):
    labels: List[str]
    weight: float


class DendrogramDrawer(ChartDrawer[LinkageTree, DendrogramStyle]):
    """
    Class to draw a `LinkageTree` as a dendrogram.

    The class has one public method `~self.draw` which draws the dendrogram.

    :param title: the title of the plot
    :param linkage_tree: the `LinkageTree` to draw
    :param style: the `DendrogramStyle` used to draw
    """

    def __init__(self, title: str, linkage_tree: LinkageTree, style: DendrogramStyle):
        super().__init__(title, linkage_tree, style)
        self._node_weight = node_weight = np.zeros(len(linkage_tree), float)

        def calculate_weights(n: Node) -> (float, int):
            """calculate the weight of a node and number of leaves under it"""
            if n.is_leaf:
                weight = n.weight
                n_leaves = 1
            else:
                l, r = linkage_tree.children(n)
                lw, ln = calculate_weights(l)
                rw, rn = calculate_weights(r)
                weight = lw + rw
                n_leaves = ln + rn
            node_weight[n.index] = weight / n_leaves
            return weight, n_leaves

        calculate_weights(linkage_tree.root)

    def _draw(self) -> None:
        """Draw the linkage tree."""
        tree_info = self._draw_node(node=self._model.root, y=0, width_relative=1.0)
        self.style.draw_leaf_labels(tree_info.labels)

    def _draw_node(self, node: Node, y: int, width_relative: float) -> _SubtreeInfo:
        """
        Recursively draw the part of the dendrogram under a node.

        :param node: the node to be drawn
        :param y: the value determining the position of the node with respect to the
          leaves of the tree
        :param width_relative: float between 0 and 1, the relative height in the tree
          of the node: the root has maximal width_relative 1
        :return info: `_SubtreeInfo` which contains weights and labels
        """
        if node.is_leaf:
            self.style.draw_link_leg(
                bottom=0.0,
                top=width_relative,
                first_leaf=y,
                n_leaves=1,
                weight=node.weight,
            )

            return _SubtreeInfo(labels=[node.label], weight=node.weight)

        else:
            child_left, child_right = self._model.children(node=node)
            if (
                self._node_weight[child_left.index]
                > self._node_weight[child_right.index]
            ):
                child_left, child_right = child_right, child_left

            info_left = self._draw_node(
                node=child_left, y=y, width_relative=node.children_distance
            )
            info_right = self._draw_node(
                node=child_right,
                y=y + len(info_left.labels),
                width_relative=node.children_distance,
            )

            info = _SubtreeInfo(
                labels=info_left.labels + info_right.labels,
                weight=info_left.weight + info_right.weight,
            )

            self.style.draw_link_connector(
                bottom=node.children_distance,
                top=width_relative,
                first_leaf=y,
                n_leaves_left=len(info_left.labels),
                n_leaves_right=len(info_right.labels),
                weight=info.weight,
            )

            return info
