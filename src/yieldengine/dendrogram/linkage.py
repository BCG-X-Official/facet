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
    def children_distance(self) -> float:
        pass

    @property
    @abstractmethod
    def weight(self) -> float:
        pass

    @property
    @abstractmethod
    def label(self) -> str:
        pass

    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    def _type_error(self, property_name: str) -> TypeError:
        return TypeError(f"{property_name} is not defined for a {type(self).__name__}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.index}"


class LinkageNode(Node):

    __slots__ = ["_children_distance"]

    def __init__(self, index: int, children_distance: Optional[float]) -> None:
        super().__init__(index=index)
        self._children_distance = children_distance

    @property
    def children_distance(self) -> float:
        return self._children_distance

    @property
    def weight(self) -> float:
        raise self._type_error("weight")

    @property
    def label(self) -> str:
        raise self._type_error("label")

    @property
    def is_leaf(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{super().__repr__()}[dist={self.children_distance * 100:.0f}%]"


class LeafNode(Node):
    __slots__ = ["_weight", "_label"]

    def __init__(self, index: int, label: str, weight: float) -> None:
        super().__init__(index=index)
        self._label = label
        self._weight = weight

    @property
    def children_distance(self) -> float:
        raise self._type_error("children_distance")

    @property
    def label(self) -> str:
        return self._label

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def is_leaf(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"{super().__repr__()}[label={self.label}, weight={self.weight}]"


class LinkageTree:

    F_CHILD_LEFT = 0
    F_CHILD_RIGHT = 1
    F_CHILDREN_DISTANCE = 2
    F_N_DESCENDANTS = 3

    def __init__(
        self,
        scipy_linkage_matrix: np.ndarray,
        leaf_labels: Iterable[str],
        leaf_weights: Iterable[float],
    ) -> None:
        # one row of the linkage matrix is a quadruple:
        # (
        #    <index of left child>,
        #    <index of right child>,
        #    <distance between children>,
        #    <number of descendant nodes, from direct children down to leaf nodes>
        # )

        def _validate_leafs(var: Sequence[Any], var_name: str):
            if len(var) != len(scipy_linkage_matrix) + 1:
                raise ValueError(
                    f"expected {len(scipy_linkage_matrix) + 1} values "
                    f"for arg {var_name}"
                )

        self._linkage_matrix = scipy_linkage_matrix
        self._leaf_labels = list(leaf_labels)
        self._leaf_weights = list(leaf_weights)

        _validate_leafs(self._leaf_labels, "leaf_labels")
        _validate_leafs(self._leaf_weights, "leaf_weights")

    def root(self) -> LinkageNode:
        return LinkageNode(
            index=len(self._linkage_matrix) * 2 - 1,
            children_distance=self._linkage_matrix[-1][LinkageTree.F_CHILDREN_DISTANCE],
        )

    @property
    def n_leaves(self) -> int:
        return len(self._leaf_labels)

    def node(self, index: int) -> Node:
        if index < self.n_leaves:
            return LeafNode(
                index=index,
                label=self._leaf_labels[index],
                weight=self._leaf_weights[index],
            )
        else:
            return LinkageNode(
                index=index,
                children_distance=self._linkage_for_node(index)[
                    LinkageTree.F_CHILDREN_DISTANCE
                ],
            )

    def children(self, node: Node) -> Optional[Tuple[Node, Node]]:
        if node.is_leaf:
            return None
        else:
            node_linkage = self._linkage_for_node(node.index)
            ix_c1, ix_c2 = node_linkage[
                [LinkageTree.F_CHILD_LEFT, LinkageTree.F_CHILD_RIGHT]
            ].astype(int)
            return self.node(ix_c1), self.node(ix_c2)

    def _linkage_for_node(self, index: int):
        return self._linkage_matrix[index - self.n_leaves]
