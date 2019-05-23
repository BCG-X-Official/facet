from typing import *

import numpy as np


class LinkageTree:
    def __init__(self, linkage_matrix: np.ndarray, leaf_labels: Sequence[str]) -> None:
        self._linkage_matrix = linkage_matrix
        self._leaf_labels = leaf_labels

    def root(self) -> "LinkageNode":
        return LinkageNode(
            index=len(self._linkage_matrix) * 2 - 1,
            label="",
            importance=1.0,
            link_distance=self._linkage_matrix[-1][2],
        )

    def direct_children(self, node: "LinkageNode") -> Optional[Iterable["LinkageNode"]]:
        if self.is_leaf(node=node):
            return None
        else:
            pass

    def all_children(self, node: "LinkageNode") -> Optional[Iterable["LinkageNode"]]:
        if self.is_leaf(node=node):
            return None
        else:
            pass

    def is_leaf(self, node: "LinkageNode") -> bool:
        return node.index < len(self._linkage_matrix)

    def feature_importance(self, node: "LinkageNode") -> float:
        # if is_leaf -> single scaled median of Shap values for the feature, in [0,1]
        # if !is_leaf -> aggregated scaled median of Shap values for features, in [0,1]
        pass

    def label(self, node: "LinkageNode") -> Optional[str]:
        if self.is_leaf(node=node):
            pass
        else:
            # only leafs have labels
            return None


class LinkageNode:
    def __init__(
        self, index: int, link_distance: Optional[float], importance: float, label: str
    ) -> None:
        self._index = index
        self._link_distance = link_distance
        self._importance = importance
        self._label = label
        pass

    @property
    def index(self) -> int:
        return self._index

    @property
    def importance(self) -> float:
        return self._importance

    @property
    def label(self) -> str:
        return self._label

    @property
    def link_distance(self) -> float:
        return self._link_distance
