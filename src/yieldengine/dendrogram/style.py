import logging
from typing import *

from matplotlib import cm
from matplotlib.axes import Axes

from yieldengine.dendrogram import DendrogramStyle

log = logging.getLogger(__name__)

RgbaColor = Tuple[float, float, float, float]


class MatplotStyle(DendrogramStyle):
    __slots__ = ["_ax"]

    def __init__(self, ax: Axes):
        super().__init__()
        self._ax = ax

    def draw_title(self, title: str) -> None:
        self._ax.set_title(label=title)

    def _draw_line(
        self, x1: float, x2: float, y1: float, y2: float, color: RgbaColor
    ) -> None:
        self._ax.plot((x1, x2), (y1, y2), color=color)


class LineStyle(MatplotStyle):
    __slots__ = ["_cm"]
    CLIP_GRADIENT_AT_WEIGHT = 0.1  # clip color gradients for weights=10

    def __init__(self, ax: Axes):
        super().__init__(ax)
        self._cm = cm.get_cmap(name="plasma", lut=256)

    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        y_axis = self._ax.yaxis
        y_axis.set_ticks(ticks=range(len(labels)))
        y_axis.set_ticklabels(ticklabels=labels)

    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        line_y = first_leaf + (n_leaves - 1) / 2
        self._draw_line(
            x1=bottom, x2=top, y1=line_y, y2=line_y, color=self.color(weight)
        )

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
            color=self.color(weight),
        )

        self.draw_link_leg(
            bottom=bottom,
            top=top,
            first_leaf=first_leaf,
            n_leaves=n_leaves_left + n_leaves_right,
            weight=weight,
        )

    def color(self, weight) -> RgbaColor:
        return self._cm(weight / LineStyle.CLIP_GRADIENT_AT_WEIGHT)
