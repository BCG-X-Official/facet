"""
Simulation drawing styles

:class:`SimulationMatplotStyle` draws some simulated low, middle and high prediction
uplift.
"""

import logging
from typing import Any, Dict, Iterable, Optional

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
    """

    def __init__(
        self,
        xticks_kwargs: Optional[Dict[str, Any]] = None,
        xlabel_kwargs: Optional[Dict[str, Any]] = None,
        xticklabels_kwargs: Optional[Dict[str, Any]] = None,
        hspace: Optional[NumericType] = None,
        xtickslabels_format: Optional[str] = None,
        figsize=(10, 10),
    ) -> None:
        super().__init__()
        self._xticks_kwargs = xticks_kwargs if xticks_kwargs else {}
        self._xlabel_kwargs = xlabel_kwargs if xlabel_kwargs else {}
        self._xticklabels_kwargs = xticklabels_kwargs if xticklabels_kwargs else {}
        self._hspace = hspace
        self._xticklabels_format = xtickslabels_format
        self._figsize = figsize

        self._color_confidence = "silver"
        self._color_middle_uplift = "red"
        # self._fig: Figure = plt.figure(figsize=self._figsize)
        self._ax_uplift_graph: Optional[Axes] = None
        self._ax_histogram: Optional[Axes] = None

    def initialize_chart(self, histogram: bool) -> None:
        """
        Initialize the chart: set figure and axes.

        :param histogram:
        :return:
        """
        if histogram:
            # self._ax_uplift_graph = self._fig.add_subplot(211)
            # self._ax_histogram = \
            #     self._fig.add_subplot(212, sharex=self._ax_uplift_graph)
            self._fig, (self._ax_uplift_graph, self._ax_histogram) = plt.subplots(
                nrows=2, figsize=self._figsize, sharex="all"
            )
        else:
            # self._ax_uplift_graph = self._fig.add_subplot(111)
            self._fig, self._ax_uplift_graph = plt.subplots(
                nrows=1, figsize=self._figsize
            )
        return None

    def draw_title(self, title: str) -> None:
        """Print the title."""
        self._fig.suptitle(title)

    def draw_uplift_graph(
        self,
        feature: str,
        target_name: str,
        feature_values: Iterable[NumericType],
        median_uplift: Iterable[NumericType],
        low_percentile_uplift: Iterable[NumericType],
        high_percentile_uplift: Iterable[NumericType],
        low_percentile: int,
        high_percentile: int,
    ):
        """
        Draw the graph with the uplift curves: median, low and high percentiles.
        """
        line_low = self.draw_low_confidence_uplift(
            feature_values, low_percentile_uplift
        )
        line_median = self.draw_middle_confidence_uplift(feature_values, median_uplift)
        line_high = self.draw_high_confidence_uplift(
            feature_values, high_percentile_uplift
        )

        labels = [
            f"{high_percentile}th percentile",
            "Median",
            f"{low_percentile}th percentile",
        ]
        handles = [line_high, line_median, line_low]
        self.draw_uplift_legend(handles=handles, labels=labels)

        xaxis_label = feature
        xticks = feature_values
        yaxis_label = f"Predicted mean uplift ({target_name})"
        self.draw_uplift_axis(
            xaxis_label=xaxis_label, yaxis_label=yaxis_label, xticks=xticks
        )

        self.draw_null_uplift_line()
        self.set_spins()

    def _draw_extreme_uplift(
        self, x: Iterable[NumericType], y: Iterable[NumericType]
    ) -> Line2D:
        # draw curve of uplift for confidence interval
        line, = self._ax_uplift_graph.plot(x, y, color=self._color_confidence)
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
        line, = self._ax_uplift_graph.plot(x, y, color=self._color_middle_uplift)
        return line

    def draw_uplift_axis(
        self, xaxis_label: str, yaxis_label: str, xticks: Iterable[NumericType]
    ):
        """
        Draw x ticks, set x labels in the uplift graph.

        :param xaxis_label: label for the x axis in the uplift graph
        :param yaxis_label:  label for the y axis in the uplift graph
        :param xticks: values used for the x ticks in the uplift graph
        """
        self._ax_uplift_graph.set_xticks(ticks=xticks, **self._xticks_kwargs)

        xticklabels = xticks
        if self._xticklabels_format:
            try:
                xticklabels = [
                    self._xticklabels_format.format(label) for label in xticklabels
                ]
            except ValueError:
                log.warning(
                    f"xticklabels_format is a wrong format type:"
                    f"{self._xticklabels_format}"
                )

        self._ax_uplift_graph.set_xticklabels(
            labels=xticklabels, **self._xticklabels_kwargs
        )

        self._ax_uplift_graph.tick_params(axis="x", labelbottom=True, bottom=False)

        self._xlabel_kwargs.update(color="black", labelpad=10, fontsize=12)
        self._ax_uplift_graph.set_xlabel(xaxis_label, **self._xlabel_kwargs)

        self._ax_uplift_graph.set_ylabel(yaxis_label, color="black", fontsize=12)

    def draw_null_uplift_line(self) -> None:
        """Draw  a horizontal line y=0 on the uplift graph."""
        self._ax_uplift_graph.axhline(y=0, color="black", linewidth=0.5)

    def set_spins(self) -> None:
        """Set the spines (data boundary lines) in the uplift graph."""
        for pos in ["top", "right", "bottom"]:
            self._ax_uplift_graph.spines[pos].set_visible(False)
        return None

    def draw_uplift_legend(
        self, handles: Iterable[Artist], labels: Iterable[str]
    ) -> None:
        """
        Draw the legend of the uplift graph.

        :param handles: the matplotlib objects to include in the legend
        :param labels: the label in the legend for the matplotlib objects in ``handles``
        """
        self._ax_uplift_graph.legend(handles, labels, frameon=False)
        return None

    def draw_histogram(
        self, feature_values, feature_frequencies, partition_width: float
    ) -> None:
        """
        Draw frequencies histogram.

        :param feature_values: the central values of the partitions
        :param feature_frequencies: the frequencies of the partitions
        :param partition_width: width of the bars in the histogram
        """

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
        plt.subplots_adjust(hspace=self._hspace)
