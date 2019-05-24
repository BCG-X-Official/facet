import logging
from typing import *

from matplotlib.axes import Axes

from yieldengine.dendrogram import DendrogramStyle, Node

log = logging.getLogger(__name__)


class MatplotStyle(DendrogramStyle):
    __slots__ = ["_ax"]

    def __init__(self, ax: Axes):
        super().__init__()
        self._ax = ax

    def draw_title(self, title: str) -> None:
        self._ax.set_title(label=title)

    def _draw_line(self, x1: float, x2: float, y1: float, y2: float, fmt: str) -> None:
        self._ax.plot((x1, x2), (y1, y2), fmt)


class LineStyle(MatplotStyle):
    def __init__(self, ax: Axes):
        super().__init__(ax)

    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        y_axis = self._ax.yaxis
        y_axis.set_ticks(ticks=range(len(labels)))
        y_axis.set_ticklabels(ticklabels=labels)

    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        line_y = first_leaf + (n_leaves - 1) / 2
        self._draw_line(x1=bottom, x2=top, y1=line_y, y2=line_y, fmt="b")

    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right,
        weight: float,
    ) -> None:
        self._draw_line(
            x1=bottom,
            x2=bottom,
            y1=first_leaf + (n_leaves_left - 1) / 2,
            y2=first_leaf + n_leaves_left + (n_leaves_right - 1) / 2,
            fmt="b",
        )

        self.draw_link_leg(
            bottom=bottom,
            top=top,
            first_leaf=first_leaf,
            n_leaves=n_leaves_left + n_leaves_right,
            weight=weight,
        )

    def color(node: Node) -> str:
        return "b"
