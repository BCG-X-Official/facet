import logging
from abc import ABC, abstractmethod
from typing import *

from yieldengine.dendrogram.linkage import LinkageTree, Node

log = logging.getLogger(__name__)


class _SubtreeInfo(NamedTuple):
    labels: List[str]
    weight: float


class DendrogramStyle(ABC):
    @abstractmethod
    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        pass

    @abstractmethod
    def draw_title(self, title: str):
        pass

    @abstractmethod
    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> int:
        pass

    @abstractmethod
    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right,
        weight: float,
    ) -> None:
        pass


class DendrogramDrawer:
    __slots__ = ["_title", "_linkage_tree", "_style"]

    def __init__(self, title: str, linkage: LinkageTree, style: DendrogramStyle):
        self._title = title
        self._linkage_tree = linkage
        self._style = style

    def draw(self) -> None:
        self._style.draw_title(self._title)
        tree_info = self._draw(node=self._linkage_tree.root(), y=0, width_relative=1.0)
        self._style.draw_leaf_labels(tree_info.labels)

    def _draw(self, node: Node, y: int, width_relative: float) -> _SubtreeInfo:
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
