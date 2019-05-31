import logging
import math
from typing import *

import matplotlib.text as mt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colorbar import make_axes, ColorbarBase
from matplotlib.colors import LogNorm
from matplotlib.ticker import Formatter

from yieldengine.dendrogram import DendrogramStyle

log = logging.getLogger(__name__)

RgbaColor = Tuple[float, float, float, float]


class _PercentageFormatter(Formatter):
    def __call__(self, x, pos=None):
        return f"{x * 100.0:.0f}%"


class MatplotStyle(DendrogramStyle):
    __slots__ = ["_ax", "_cm", "_min_weight"]

    _PERCENTAGE_FORMATTER = _PercentageFormatter()

    def __init__(self, ax: Axes, min_weight: float = 0.01) -> None:
        super().__init__()
        self._ax = ax

        if min_weight > 1.0 or min_weight <= 0.0:
            raise ValueError("arg min_weight must be > 0.0 and <= 1.0")

        self._min_weight = min_weight

        ax.ticklabel_format(axis="x", scilimits=(-3, 3))

        cax, _ = make_axes(ax)
        self._cm = cm.get_cmap(name="plasma", lut=256)
        self._cb = ColorbarBase(
            cax,
            cmap=self._cm,
            norm=LogNorm(min_weight, 1),
            label="feature importance",
            orientation="vertical",
        )

        cax.yaxis.set_minor_formatter(MatplotStyle._PERCENTAGE_FORMATTER)
        cax.yaxis.set_major_formatter(MatplotStyle._PERCENTAGE_FORMATTER)

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
    def __init__(self, ax: Axes, min_weight: float = 0.01) -> None:
        super().__init__(ax=ax, min_weight=min_weight)
        ax.margins(0, 0)

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
        fill_color = self.color(weight)

        self._ax.barh(
            y=[y - 0.5],
            width=[w],
            height=[h],
            left=[x],
            align="edge",
            color=fill_color,
            edgecolor="white",
            linewidth=1,
        )

        weight_percent = weight * 100
        label = (
            f"{weight_percent :.2g}%"
            if round(weight_percent, 1) < 100
            else f"{weight_percent :.3g}%"
        )
        fig = self._ax.figure
        (x0, _), (x1, _) = self._ax.transData.inverted().transform(
            mt.Text(0, 0, label, figure=fig).get_window_extent(
                fig.canvas.get_renderer()
            )
        )

        if abs(x1 - x0) <= w:
            fill_luminance = sum(fill_color[:3]) / 3
            text_color = "white" if fill_luminance < 0.5 else "black"
            self._ax.text(
                x + w / 2,
                y + (h - 1) / 2,
                label,
                ha="center",
                va="center",
                color=text_color,
            )
