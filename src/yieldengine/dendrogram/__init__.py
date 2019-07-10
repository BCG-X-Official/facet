"""Specify the high-level mechanisms to draw a dendrogram.

The class :class:`~DendrogramDrawer` draws a dendrogram based
on a :class:`~.linkage.LinkageTree` and a :class:`~DendrogramStyle`.
The class `~DendrogramStyle` is an abstract class that that must be implemented for
each specific style.
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np

from yieldengine.dendrogram.linkage import LinkageTree, Node

log = logging.getLogger(__name__)


class _SubtreeInfo(NamedTuple):
    labels: List[str]
    weight: float


class DendrogramStyle(ABC):
    """
    Base class for dendrogram drawing styles.

    Implementations must define `draw_leaf_labels`, `draw_title`, `draw_link_leg` \
    and `draw_link_connector`.
    """

    @abstractmethod
    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        """Render the labels for all leaves."""
        pass

    @abstractmethod
    def draw_title(self, title: str) -> None:
        """Draw the title of the dendrogram."""
        pass

    @abstractmethod
    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        """Draw a leaf of the linkage tree.

        :param bottom: the clustering level (i.e. similarity) of the child nodes
        :param top: the clustering level (i.e. similarity) of the parent node
        :param first_leaf: the index of the first leaf in the tree
        :param n_leaves: number of leaves under consideration, always set to 1
        :param weight: the weight of the parent node
        """
        pass

    @abstractmethod
    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right: int,
        weight: float,
    ) -> None:
        """Draw a connector between two child nodes and their parent node.

        :param bottom: the clustering level (i.e. similarity) of the child nodes
        :param top: the clustering level (i.e. similarity) of the parent node
        :param first_leaf: the index of the first leaf in the tree
        :param n_leaves_left: the number of leaves in the left sub-tree
        :param n_leaves_right: the number of leaves in the right sub-tree
        :param weight: the weight of the parent node
        """
        pass


class DendrogramDrawer:
    """Class to draw a `LinkageTree` as a dendrogram.

    :param title: the title of the plot
    :param linkage_tree: the `LinkageTree` to draw
    :param style: the `DendrogramStyle` used to draw
    """

    __slots__ = ["_title", "_linkage_tree", "_style", "_node_weight"]

    def __init__(self, title: str, linkage_tree: LinkageTree, style: DendrogramStyle):
        self._title = title
        self._linkage_tree = linkage_tree
        self._style = style
        self._node_weight = node_weight = np.zeros(len(linkage_tree), float)

        def calculate_weights(n: Node) -> (float, int):
            """
            Calculate the weight of a node and number of leaves under it.

            :param n: the node
            :return: tuple (weight, n_leaves)
            """
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

    def draw(self) -> None:
        """Draw the linkage tree."""
        self._style.draw_title(self._title)
        tree_info = self._draw(node=self._linkage_tree.root, y=0, width_relative=1.0)
        self._style.draw_leaf_labels(tree_info.labels)

    def _draw(self, node: Node, y: int, width_relative: float) -> _SubtreeInfo:
        """Recursively draw the part of the dendrogram for a given node and its
        children.

        :param node: the node to be drawn
        :param y: the value determining the position of the node with respect to the
          leaves of the tree
        :param width_relative: float between 0 and 1, the relative height in the tree
          of the node: the root has maximal width_relative 1
        :return info: `_SubtreeInfo` which contains weights and labels
        """
        if node.is_leaf:
            self._style.draw_link_leg(
                bottom=0.0,
                top=width_relative,
                first_leaf=y,
                n_leaves=1,
                weight=node.weight,
            )

            return _SubtreeInfo(labels=[node.label], weight=node.weight)

        else:
            child_left, child_right = self._linkage_tree.children(node=node)
            if (
                self._node_weight[child_left.index]
                > self._node_weight[child_right.index]
            ):
                child_left, child_right = child_right, child_left

            info_left = self._draw(
                node=child_left, y=y, width_relative=node.children_distance
            )
            info_right = self._draw(
                node=child_right,
                y=y + len(info_left.labels),
                width_relative=node.children_distance,
            )

            info = _SubtreeInfo(
                labels=info_left.labels + info_right.labels,
                weight=info_left.weight + info_right.weight,
            )

            self._style.draw_link_connector(
                bottom=node.children_distance,
                top=width_relative,
                first_leaf=y,
                n_leaves_left=len(info_left.labels),
                n_leaves_right=len(info_right.labels),
                weight=info.weight,
            )

            return info
