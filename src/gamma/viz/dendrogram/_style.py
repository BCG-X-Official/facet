"""
Dendrogram styles.

The dendrogram styles are given as a parameter to a
:class:`~yieldengine.dendrogram.DendrogramDrawer` and determine the style of the plot.

:class:`DendrogramMatplotStyle` is a an abstract base class for styles using matplotlib.

:class:`~LineStyle` renders dendrogram trees in the classical style as a line drawing.

:class:`~FeatMapStyle` renders dendrogram trees as a combination of tree and heatmap
for better visibility of feature importance.
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Sequence, TextIO

from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase, make_axes
from matplotlib.colors import LogNorm
from matplotlib.ticker import Formatter

from gamma.viz import ChartStyle, MatplotStyle, RgbaColor, TextStyle
from gamma.viz.text import CharacterMatrix

log = logging.getLogger(__name__)

_COLOR_BLACK = "black"
_COLOR_WHITE = "white"


class _PercentageFormatter(Formatter):
    """Format percentage."""

    def __call__(self, x, pos=None) -> str:
        return f"{x * 100.0:.0f}%"


class DendrogramStyle(ChartStyle, ABC):
    """
    Base class for dendrogram drawing styles.

    Implementations must define `draw_leaf_labels`, `draw_title`, `draw_link_leg` \
    and `draw_link_connector`.
    """

    @abstractmethod
    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        """Render the labels for all leaves.

        :param labels: labels of the leaves
        """
        pass

    @abstractmethod
    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        """
        Draw a leaf of the linkage tree.

        :param bottom: the x coordinate of the child node
        :param top: the x coordinate of the parent node
        :param first_leaf: the index of the first leaf in the current sub-tree
        :param n_leaves: the number of leaves in the current sub-tree
        :param weight: the weight of the child node
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
        """
        Draw a connector between two child nodes and their parent node.

        :param bottom: the clustering level (i.e. similarity) of the child nodes
        :param top: the clustering level (i.e. similarity) of the parent node
        :param first_leaf: the index of the first leaf in the left sub-tree
        :param n_leaves_left: the number of leaves in the left sub-tree
        :param n_leaves_right: the number of leaves in the right sub-tree
        :param weight: the weight of the parent node
        """
        pass


class DendrogramMatplotStyle(DendrogramStyle, MatplotStyle, ABC):
    """
    Base class for Matplotlib styles for dendrogram.

    Provide basic support for plotting a color legend for feature importance,
    and providing the `Axes` object for plotting the actual dendrogram including
    tick marks for the feature distance axis.

    :param ax: :class:`matplotlib.axes.Axes` object to draw on
    :param min_weight: the min weight on the feature importance color scale, must be \
                       greater than 0 and smaller than 1 (default: 0.01)
    """

    _PERCENTAGE_FORMATTER = _PercentageFormatter()

    def __init__(self, ax: Axes, min_weight: float = 0.01) -> None:
        super().__init__(ax=ax)

        if min_weight >= 1.0 or min_weight <= 0.0:
            raise ValueError("arg min_weight must be > 0.0 and < 1.0")

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

        cax.yaxis.set_minor_formatter(DendrogramMatplotStyle._PERCENTAGE_FORMATTER)
        cax.yaxis.set_major_formatter(DendrogramMatplotStyle._PERCENTAGE_FORMATTER)

    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        """Draw leaf labels on the dendrogram."""
        y_axis = self.ax.yaxis
        y_axis.set_ticks(ticks=range(len(labels)))
        y_axis.set_ticklabels(ticklabels=labels)

    def color(self, weight: float) -> RgbaColor:
        """
        Return the color associated to the feature weight (= importance).

        :param weight: the weight
        :return: the color as a RGBA tuple
        """
        return self._cm(
            0
            if weight <= self._min_weight
            else 1 - math.log(weight) / math.log(self._min_weight)
        )


class LineStyle(DendrogramMatplotStyle):
    """The classical dendrogram style, as a coloured tree diagram."""

    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        """
        Draw a horizontal link in the dendrogram between a node and one of its children.

        See :func:`~yieldengine.dendrogram.DendrogramStyle.draw_link_leg` for the
        documentation of the abstract method.

        :param bottom: the x coordinate of the child node
        :param top: the x coordinate of the parent node
        :param first_leaf: the index of the first leaf in the current sub-tree
        :param n_leaves: the number of leaves in the current sub-tree
        :param weight: the weight of the child node
        """
        line_y = first_leaf + (n_leaves - 1) / 2
        self._draw_line(x1=bottom, x2=top, y1=line_y, y2=line_y, weight=weight)

    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right: int,
        weight: float,
    ) -> None:
        """
        Draw a vertical link between two sibling nodes and the outgoing vertical line.

        See :func:`~yieldengine.dendrogram.DendrogramStyle.draw_link_connector` for the
        documentation of the abstract method.

        :param bottom: the clustering level (i.e. similarity) of the child nodes
        :param top: the clustering level (i.e. similarity) of the parent node
        :param first_leaf: the index of the first leaf in the left sub-tree
        :param n_leaves_left: the number of leaves in the left sub-tree
        :param n_leaves_right: the number of leaves in the right sub-tree
        :param weight: the weight of the parent node
        """
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
        self.ax.plot((x1, x2), (y1, y2), color=self.color(weight))


class FeatMapStyle(DendrogramMatplotStyle):
    """
    Plot dendrograms with a heat map style.

    :param ax: a matplotlib `Axes`
    :param min_weight: the min weight in the color bar
    """

    def __init__(self, ax: Axes, min_weight: float = 0.01) -> None:
        super().__init__(ax=ax, min_weight=min_weight)
        ax.margins(0, 0)
        ax.set_xlim(0, 1)

    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        """
        Draw a horizontal box in the dendrogram for a leaf.

        See :func:`~yieldengine.dendrogram.DendrogramStyle.draw_link_leg` for the
        documentation of the abstract method.

        :param bottom: the x coordinate of the child node
        :param top: the x coordinate of the parent node
        :param first_leaf: the index of the first leaf in the current sub-tree
        :param n_leaves: the number of leaves in the current sub-tree
        :param weight: the weight of the child node
        """
        line_y = first_leaf + (n_leaves - 1) / 2
        self._draw_hbar(x=bottom, w=top - bottom, y=line_y, h=1, weight=weight)

    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right: int,
        weight: float,
    ) -> None:
        """
        Draw a link between a node and its two children as a box.

        See :func:`~yieldengine.dendrogram.DendrogramStyle.draw_link_connector` for the
        documentation of the abstract method.

        :param bottom: the clustering level (i.e. similarity) of the child nodes
        :param top: the clustering level (i.e. similarity) of the parent node
        :param first_leaf: the index of the first leaf in the left sub-tree
        :param n_leaves_left: the number of leaves in the left sub-tree
        :param n_leaves_right: the number of leaves in the right sub-tree
        :param weight: the weight of the parent node
        """
        self._draw_hbar(
            x=bottom,
            w=top - bottom,
            y=first_leaf,
            h=n_leaves_left + n_leaves_right,
            weight=weight,
        )

    def _draw_hbar(self, x: float, y: float, w: float, h: float, weight: float) -> None:
        """
        Draw a box.

        :param x: left x position of the box
        :param y: top vertical position of the box
        :param w: the width of the box
        :param h: the height of the box
        :param weight: the weight used to compute the color of the box
        """
        fill_color = self.color(weight)

        self.ax.barh(
            y=[y - 0.5],
            width=[w],
            height=[h],
            left=[x],
            align="edge",
            color=fill_color,
            edgecolor=_COLOR_WHITE,
            linewidth=1,
        )

        weight_percent = weight * 100
        label = (
            f"{weight_percent:.2g}%"
            if round(weight_percent, 1) < 100
            else f"{weight_percent:.3g}%"
        )

        x_text = x + w / 2
        y_text = y + (h - 1) / 2
        text_width, _ = self.text_size(text=label, x=x_text, y=y_text)

        if text_width <= w:
            fill_luminance = sum(fill_color[:3]) / 3
            text_color = _COLOR_WHITE if fill_luminance < 0.5 else _COLOR_BLACK
            self.ax.text(
                x_text, y_text, label, ha="center", va="center", color=text_color
            )


class DendrogramTextStyle(DendrogramStyle, TextStyle):
    """
    Dendrogram rendered as text.
    """

    def __init__(
        self, out: TextIO = None, width: int = 80, max_height: int = 100
    ) -> None:
        super().__init__(out, width)
        if max_height <= 0:
            raise ValueError(
                f"arg max_height expected to be positive integer but is {max_height}"
            )
        self._max_height = max_height
        self._dendrogram_left = min(20, width // 2)
        self._char_matrix = None
        self._n_labels = None

    def drawing_start(self, title: str) -> None:
        """Write the title."""
        self.out.write(f"{title:*^{self.width}s}\n")
        self._char_matrix = CharacterMatrix(
            n_rows=self._max_height, n_columns=self.width
        )

    def drawing_finalize(self) -> None:
        """Finalize writing the text."""
        try:
            super().drawing_finalize()
            for row in reversed(range(self._n_labels + 1)):
                self.out.write(f"{self._char_matrix[row, :]}\n")
        finally:
            self._char_matrix = None
            self._n_labels = None

    def draw_leaf_labels(self, labels: Sequence[str]) -> None:
        """
        Draw the feature labels in the drawing.
        
        :param labels: the name of the features 
        """
        matrix = self._char_matrix
        n_labels = len(labels)
        if n_labels > self._max_height:
            n_labels = self._max_height - 1
            matrix[n_labels, :] = f"{'clipped╮':~^{self.width}s}\n"
        self._n_labels = n_labels
        label_width = self._weight_column
        for row, label in enumerate(labels[:n_labels]):
            matrix[row, :label_width] = label

    @property
    def _weight_column(self) -> int:
        return self._dendrogram_left - 5

    def draw_link_leg(
        self, bottom: float, top: float, first_leaf: int, n_leaves: int, weight: float
    ) -> None:
        """
        Draw a horizontal link in the dendrogram between a node and one of its children.

        :param bottom: the x coordinate of the child node
        :param top: the x coordinate of the parent node
        :param first_leaf: the index of the first leaf in the current sub-tree
        :param n_leaves: the number of leaves in the current sub-tree
        :param weight: the weight of the child node
        """
        line_y = first_leaf + (n_leaves - 1) // 2
        self._char_matrix[line_y, self._x_pos(bottom) : self._x_pos(top) + 1] = "-"
        if n_leaves == 1:
            # we're in a leaf so can draw the weight next to he label
            self._char_matrix[
                line_y, self._weight_column : self._dendrogram_left
            ] = f"{weight * 100:3.0f}%"

    def draw_link_connector(
        self,
        bottom: float,
        top: float,
        first_leaf: int,
        n_leaves_left: int,
        n_leaves_right: int,
        weight: float,
    ) -> None:
        """
        Draw a vertical link between two sibling nodes and the outgoing vertical line.

        See :func:`~yieldengine.dendrogram.DendrogramStyle.draw_link_connector` for the
        documentation of the abstract method.

        :param bottom: the clustering level (i.e. similarity) of the child nodes
        :param top: the clustering level (i.e. similarity) of the parent node
        :param first_leaf: the index of the first leaf in the left sub-tree
        :param n_leaves_left: the number of leaves in the left sub-tree
        :param n_leaves_right: the number of leaves in the right sub-tree
        :param weight: the weight of the parent node
        """
        self.draw_link_leg(
            bottom=bottom,
            top=top,
            first_leaf=first_leaf,
            n_leaves=n_leaves_left + n_leaves_right,
            weight=weight,
        )

        y1 = first_leaf + (n_leaves_left - 1) // 2
        y2 = first_leaf + n_leaves_left + (n_leaves_right - 1) // 2
        x = self._x_pos(bottom)
        matrix = self._char_matrix
        if y2 - y1 > 1:
            matrix[(y1 + 1) : y2, x] = "|"
        matrix[y1, x] = "/"
        matrix[y2, x] = "\\"

    def _x_pos(self, h: float) -> int:
        return self._dendrogram_left + int((self.width - self._dendrogram_left) * h)
