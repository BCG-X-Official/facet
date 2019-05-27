import logging
import math
from typing import *

from matplotlib import cm
from matplotlib.axes import Axes

from yieldengine.dendrogram import DendrogramStyle

log = logging.getLogger(__name__)

RgbaColor = Tuple[float, float, float, float]


class MatplotStyle(DendrogramStyle):
    __slots__ = ["_ax", "_cm", "_min_weight"]

    def __init__(self, ax: Axes, min_weight: float = 0.01) -> None:
        super().__init__()
        self._ax = ax

        ax.ticklabel_format(axis="x", scilimits=(-3, 3))
        self._cm = cm.get_cmap(name="plasma", lut=256)
        self._min_weight = min_weight

    def draw_title(self, title: str) -> None:
        self._ax.set_title(label=title)

    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        y_axis = self._ax.yaxis
        y_axis.set_ticks(ticks=range(len(labels)))
        y_axis.set_ticklabels(ticklabels=labels)

    def color(self, weight) -> RgbaColor:
        return self._cm(
            0
            if weight <= self._min_weight
            else 1 - math.log(weight) / math.log(self._min_weight)
        )


class LineStyle(MatplotStyle):
    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        line_y = first_leaf + (n_leaves - 1) / 2
        self._draw_line(x1=bottom, x2=top, y1=line_y, y2=line_y, weight=weight)

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
            weight=weight,
        )

        self.draw_link_leg(
            bottom=bottom,
            top=top,
            first_leaf=first_leaf,
            n_leaves=n_leaves_left + n_leaves_right,
            weight=weight,
        )

    def _draw_line(
        self, x1: float, x2: float, y1: float, y2: float, weight: float
    ) -> None:
        self._ax.plot((x1, x2), (y1, y2), color=self.color(weight))


class FeatMapStyle(MatplotStyle):
    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        line_y = first_leaf + (n_leaves - 1) / 2
        self._draw_hbar(x=bottom, w=top - bottom, y=line_y, h=1, weight=weight)

    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right,
        weight: float,
    ) -> None:
        self._draw_hbar(
            x=bottom,
            w=top - bottom,
            y=first_leaf,
            h=n_leaves_left + n_leaves_right,
            weight=weight,
        )

    def _draw_hbar(self, x: float, y: float, w: float, h: float, weight: float) -> None:
        self._ax.barh(
            y=[y - 0.5],
            width=[w],
            height=[h],
            left=[x],
            align="edge",
            color=self.color(weight),
            edgecolor="white",
            linewidth=1,
        )
