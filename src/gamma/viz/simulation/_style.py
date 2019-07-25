"""
Simulation drawing styles

:class:`SimulationMatplotStyle` draws some simulated low, middle and high prediction
uplift.
"""

import logging
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from gamma.viz import MatplotStyle
from gamma.yieldengine.partition import T_NumericValue

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
    """

    def __init__(self, ax: Axes) -> None:
        super().__init__()
        self._ax_uplift_graph: Axes = ax
        self._ax_histogram: Optional[Axes] = None

        self._color_confidence = "silver"
        self._color_middle_uplift = "red"

    @property
    def ax_uplift_graph(self) -> Axes:
        return self._ax_uplift_graph

    @property
    def ax_histogram(self) -> Optional[Axes]:
        return self._ax_histogram

    def draw_title(self, title: str) -> None:
        """Print the title."""
        self._ax_uplift_graph.figure.suptitle(title)

    def draw_uplift_graph(
        self,
        feature: str,
        target_name: str,
        feature_values: Iterable[T_NumericValue],
        median_uplift: Iterable[T_NumericValue],
        low_percentile_uplift: Iterable[T_NumericValue],
        high_percentile_uplift: Iterable[T_NumericValue],
        low_percentile: int,
        high_percentile: int,
    ):
        """
        Draw the graph with the uplift curves: median, low and high percentiles.
        """
        line_low = self._draw_low_confidence_uplift(
            feature_values, low_percentile_uplift
        )
        line_median = self._draw_middle_confidence_uplift(feature_values, median_uplift)
        line_high = self._draw_high_confidence_uplift(
            feature_values, high_percentile_uplift
        )

        labels = [
            f"{high_percentile}th percentile",
            "Median",
            f"{low_percentile}th percentile",
        ]
        handles = [line_high, line_median, line_low]
        self._draw_uplift_legend(handles=handles, labels=labels)

        xaxis_label = feature
        xticks = feature_values
        yaxis_label = f"Predicted mean uplift ({target_name})"
        self._draw_uplift_axis(
            xaxis_label=xaxis_label, yaxis_label=yaxis_label, xticks=xticks
        )

        self._draw_null_uplift_line()
        self._set_spins()

    def draw_histogram(
        self, feature_values, feature_frequencies, partition_width: float
    ) -> None:
        """
        Draw frequencies histogram.

        :param feature_values: the central values of the partitions
        :param feature_frequencies: the frequencies of the partitions
        :param partition_width: width of the bars in the histogram
        """
        self._ax_histogram = self._ax_uplift_graph.figure.add_subplot(212)

        self._ax_histogram.bar(
            x=feature_values,
            height=feature_frequencies,
            width=partition_width,
            color="silver",
            edgecolor="white",
        )
        self._ax_histogram.invert_yaxis()
        self._ax_histogram.tick_params(axis="y", labelcolor="black")
        max_y = max(feature_frequencies)
        y_offset = max_y * 0.05
        for (_x, _y) in zip(feature_values, feature_frequencies):
            if _y > 0:
                self._ax_histogram.text(
                    _x,
                    _y + y_offset,
                    str(_y),
                    color="black",
                    horizontalalignment="center",
                )
        self._ax_histogram.get_yaxis().set_visible(False)
        self._ax_histogram.get_xaxis().set_visible(False)
        for pos in ["top", "right", "left", "bottom"]:
            self._ax_histogram.spines[pos].set_visible(False)
        plt.subplots_adjust(hspace=0.2)

    def _draw_extreme_uplift(
        self, x: Iterable[T_NumericValue], y: Iterable[T_NumericValue]
    ) -> Line2D:
        # draw curve of uplift for confidence interval
        line, = self._ax_uplift_graph.plot(x, y, color=self._color_confidence)
        return line

    def _draw_low_confidence_uplift(
        self, x: Iterable[T_NumericValue], y: Iterable[T_NumericValue]
    ) -> Line2D:
        """Draw the low confidence uplift curve.

        :param x: simulated feature values
        :param y: low confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self._draw_extreme_uplift(x, y)

    def _draw_high_confidence_uplift(
        self, x: Iterable[T_NumericValue], y: Iterable[T_NumericValue]
    ) -> Line2D:
        """
        Draw the high confidence uplift curve.

        :param x: simulated feature values
        :param y: high confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self._draw_extreme_uplift(x, y)

    def _draw_middle_confidence_uplift(
        self, x: Iterable[T_NumericValue], y: Iterable[T_NumericValue]
    ) -> Line2D:
        """
        Draw the middle uplift curve.

        :param x: simulated feature values
        :param y: middle confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        line, = self._ax_uplift_graph.plot(x, y, color=self._color_middle_uplift)
        return line

    def _draw_uplift_axis(
        self, xaxis_label: str, yaxis_label: str, xticks: Iterable[T_NumericValue]
    ) -> None:
        """
        Draw x ticks, set x labels in the uplift graph.

        :param xaxis_label: label for the x axis in the uplift graph
        :param yaxis_label:  label for the y axis in the uplift graph
        :param xticks: values used for the x ticks in the uplift graph
        """
        self._ax_uplift_graph.set_xticks(ticks=xticks)

        self._ax_uplift_graph.set_xticklabels(labels=xticks)

        self._ax_uplift_graph.tick_params(axis="x", labelbottom=True, bottom=False)

        self._ax_uplift_graph.set_xlabel(xaxis_label)

        self._ax_uplift_graph.set_ylabel(yaxis_label, color="black", fontsize=12)
        return None

    def _draw_null_uplift_line(self) -> None:
        """Draw  a horizontal line y=0 on the uplift graph."""
        self._ax_uplift_graph.axhline(y=0, color="black", linewidth=0.5)

    def _set_spins(self) -> None:
        """Set the spines (data boundary lines) in the uplift graph."""
        for pos in ["top", "right", "bottom"]:
            self._ax_uplift_graph.spines[pos].set_visible(False)
        return None

    def _draw_uplift_legend(
        self, handles: Iterable[Artist], labels: Iterable[str]
    ) -> None:
        """
        Draw the legend of the uplift graph.

        :param handles: the matplotlib objects to include in the legend
        :param labels: the label in the legend for the matplotlib objects in ``handles``
        """
        self._ax_uplift_graph.legend(handles, labels, frameon=False)
        return None
