from typing import NamedTuple

from yieldengine.feature.linkage import LeafNode, LinkageNode, LinkageTree, Node


class _SubtreeInfo(NamedTuple):
    height: int
    weight: float


class DendrogramDrawer:
    # note: draws for left->right oriented version (i.e. root node on the right)
    def __init__(self, linkage: LinkageTree):
        self._linkage_tree = linkage

    def draw(self, width: int, height: int) -> None:
        # initialize figure/canvas using supplied bounds
        pass

        # run _draw_dendrogram
        self._draw_dendrogram(node=self._linkage_tree.root(), y=0, width_relative=1.0)

    def _draw_dendrogram(
        self, node: Node, y: int, width_relative: float
    ) -> _SubtreeInfo:
        # returns height
        if isinstance(node, LeafNode):
            height = self._draw_leaf_label(y=y, label=node.label)

            self._draw_link_leg(
                x1_relative=0.0,
                x2_relative=width_relative,
                y=y + height // 2,
                weight=node.weight,
            )

            return _SubtreeInfo(height=height, weight=node.weight)

        elif isinstance(node, LinkageNode):
            child_left, child_right = self._linkage_tree.children(node=node)
            info_left = self._draw_dendrogram(
                node=child_left, y=y, width_relative=node.children_distance
            )
            info_right = self._draw_dendrogram(
                node=child_right,
                y=y + info_left.height,
                width_relative=node.children_distance,
            )

            info = _SubtreeInfo(
                height=info_left.height + info_right.height,
                weight=info_left.weight + info_right.weight,
            )

            self._draw_link_connector(
                x1_relative=node.children_distance,
                x2_relative=width_relative,
                y=y,
                info_left=info_left,
                info_right=info_right,
            )

            return info

        else:
            raise TypeError(f"unknown node type: {type(node)}")

    def _draw_leaf_label(self, y: int, label: str) -> int:
        pass

    def _draw_link_leg(
        self, x1_relative: float, x2_relative: float, y: int, weight: float
    ) -> int:
        # get colour by DendrogramDrawer.color(node)
        # draw coloured line from x=[0,node.link_distance] on y=y
        # returns height
        pass

    def _draw_link_connector(
        self,
        x1_relative: float,
        x2_relative: float,
        y: int,
        info_left: _SubtreeInfo,
        info_right: _SubtreeInfo,
    ) -> None:

        self._draw_link_leg(
            x1_relative=x1_relative,
            x2_relative=x2_relative,
            y=int((y + info_left.height + info_right.weight) / 2),
            weight=info_left.weight + info_right.weight,
        )
        pass

    @staticmethod
    def color(node: LinkageNode) -> str:
        # map node.importance to hex-string
        pass
