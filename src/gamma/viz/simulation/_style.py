"""
Simulation drawing styles

:class:`SimulationMatplotStyle` draws some simulated low, middle and high prediction
uplift.
"""

import logging
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from gamma.viz import MatplotStyle
from gamma.yieldengine.partition import NumericType

log = logging.getLogger(__name__)


class SimulationMatplotStyle(MatplotStyle):
    """
    Matplotlib Style for simulation chart.

    Allows to plot two different graph:

    - an uplift graph that shows on the x axis the simulated feature values,
      and on the y axis the uplift prediction under the assumption that the simulated
      feature takes the value given on the x axis. There are three curves on the
      graph: for low, middle and high confidence

    - a histogram of the feature simulated values

    :param ax: the matplotlib axes which contains the chart(s)
    """

    def __init__(self, ax: Optional[Axes] = None):
        super().__init__(ax=ax)
        self._color_confidence = "silver"
        self._color_middle_uplift = "red"
        self._ax = ax

    @property
    def ax(self) -> Axes:
        """The current ax of the chart."""
        return self._ax

    @ax.setter
    def ax(self, new_ax: Axes):
        self._ax = new_ax

    def draw_title(self, title: str) -> None:
        """Print the title."""
        pass

    def _draw_extreme_uplift(
        self, x: Iterable[NumericType], y: Iterable[NumericType]
    ) -> Line2D:
        # draw curve of uplift for confidence interval
        line, = self._ax.plot(x, y, color=self._color_confidence)
        return line

    def draw_low_confidence_uplift(
        self, x: Iterable[NumericType], y: Iterable[NumericType]
    ) -> Line2D:
        """Draw the low confidence uplift curve.

        :param x: simulated feature values
        :param y: low confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self._draw_extreme_uplift(x, y)

    def draw_high_confidence_uplift(
        self, x: Iterable[NumericType], y: Iterable[NumericType]
    ) -> Line2D:
        """
        Draw the high confidence uplift curve.

        :param x: simulated feature values
        :param y: high confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self._draw_extreme_uplift(x, y)

    def draw_middle_confidence_uplift(
        self, x: Iterable[NumericType], y: Iterable[NumericType]
    ) -> Line2D:
        """
        Draw the middle uplift curve.

        :param x: simulated feature values
        :param y: middle confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        line, = self._ax.plot(x, y, color=self._color_middle_uplift)
        return line

    def draw_uplift_axis(
        self, xaxis_label: str, yaxis_label: str, x_ticks: List[NumericType]
    ):
        """
        Draw x ticks, set x labels in the uplift graph.

        :param xaxis_label: label for the x axis in the uplift graph
        :param yaxis_label:  label for the y axis in the uplift graph
        :param x_ticks: values used for the x ticks in the uplift graph
        """
        self.ax.set_xlabel(xaxis_label, color="black", labelpad=10, fontsize=12)
        self.ax.set_ylabel(yaxis_label, color="black", fontsize=12)
        self.ax.set_xticks(ticks=x_ticks)
        self.ax.tick_params(axis="x", labelbottom=True, bottom=False)

    def draw_null_uplift_line(self) -> None:
        """Draw  a horizontal line y=0 on the uplift graph."""
        self.ax.axhline(y=0, color="black", linewidth=0.5)

    def set_spins(self) -> None:
        """Set the spines (data boundary lines) in the uplift graph."""
        for pos in ["top", "right", "bottom"]:
            self.ax.spines[pos].set_visible(False)
        return None

    def draw_uplift_legend(self, handles: List[Artist], labels: List[str]) -> None:
        """
        Draw the legend of the uplift graph.

        :param handles: the matplotlib objects to include in the legend
        :param labels: the label in the legend for the matplotlib objects in ``handles``
        """
        self.ax.legend(handles, labels, frameon=False)
        return None

    def draw_histogram(self, x, height, width: float) -> None:
        """
        Draw frequencies histogram.

        :param x: the central values of the partitions
        :param height: the frequencies of the partitions
        :param width: width of the bars in the histogram
        """

        self.ax.bar(
            x=x, height=height, width=0.9 * width, color="silver", edgecolor="white"
        )
        self.ax.invert_yaxis()
        self.ax.tick_params(axis="y", labelcolor="black")
        max_y = max(height)
        y_offset = max_y * 0.05
        for (_x, _y) in zip(x, height):
            if _y > 0:
                self.ax.text(
                    _x,
                    _y + y_offset,
                    str(_y),
                    color="black",
                    horizontalalignment="center",
                )
        self.ax.get_yaxis().set_visible(False)
        self.ax.get_xaxis().set_visible(False)
        for pos in ["top", "right", "left", "bottom"]:
            self.ax.spines[pos].set_visible(False)
        plt.subplots_adjust(hspace=0.2)
