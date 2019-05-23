from abc import ABC, abstractmethod
from typing import *

import numpy as np


class Node(ABC):
    def __init__(self, index: int) -> None:
        if type(self) == Node:
            raise TypeError("cannot instantiate abstract class Node")
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        pass


class LinkageNode(Node):
    __slots__ = ["_children_distance"]

    def __init__(self, index: int, children_distance: Optional[float]) -> None:
        super().__init__(index=index)
        self._children_distance = children_distance

    @property
    def children_distance(self) -> float:
        return self._children_distance

    @property
    def is_leaf(self) -> bool:
        return False


class LeafNode(Node):
    __slots__ = ["_weight", "_label"]

    def __init__(self, index: int, weight: float, label: str) -> None:
        super().__init__(index=index)
        self._weight = weight
        self._label = label

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_leaf(self) -> bool:
        return True


class LinkageTree:

    F_CHILD_LEFT = 0
    F_CHILD_RIGHT = 1
    F_CHILDREN_DISTANCE = 2
    F_N_DESCENDANTS = 3

    def __init__(
        self,
        scipy_linkage_matrix: np.ndarray,
        leaf_labels: Sequence[str],
        leaf_weights: Sequence[float],
    ) -> None:
        # one row of the linkage matrix is a quadruple:
        # (
        #    <index of left child>,
        #    <index of right child>,
        #    <distance between children>,
        #    <number of descendant nodes, from direct children down to leaf nodes>
        # )
        self._linkage_matrix = scipy_linkage_matrix
        self._leaf_labels = leaf_labels
        self._leaf_weights = leaf_weights

    def root(self) -> LinkageNode:
        return LinkageNode(
            index=len(self._linkage_matrix) * 2 - 1,
            children_distance=self._linkage_matrix[-1][LinkageTree.F_CHILDREN_DISTANCE],
        )

    def _linkage_for_node(self, index: int):
        return self._linkage_matrix[index - len(self._linkage_matrix)]

    def node(self, index: int) -> Node:
        if index < len(self._linkage_matrix):
            return LeafNode(
                index=index,
                weight=self._leaf_weights[index],
                label=self._leaf_labels[index],
            )
        else:
            return LinkageNode(
                index=index,
                children_distance=self._linkage_for_node(index)[
                    LinkageTree.F_CHILDREN_DISTANCE
                ],
            )

    def children(self, node: LinkageNode) -> Optional[Tuple[LinkageNode, LinkageNode]]:
        if node.is_leaf:
            return None
        else:
            node_linkage = self._linkage_for_node(node.index)
            return node_linkage[[LinkageTree.F_CHILD_LEFT, LinkageTree.F_CHILD_RIGHT]]
