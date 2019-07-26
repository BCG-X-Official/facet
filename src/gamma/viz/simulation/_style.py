"""
Simulation drawing styles

:class:`SimulationMatplotStyle` draws some simulated low, middle and high prediction
uplift.
"""

import logging
from typing import Iterable, Optional

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_size import Scaled

from gamma import ListLike
from gamma.viz import MatplotStyle
from gamma.yieldengine.partition import Partitioning, T_Number

log = logging.getLogger(__name__)


class SimulationMatplotStyle(MatplotStyle):
    """
    Matplotlib Style for simulation chart.

    Allows to plot two different graph:

    - an uplift graph that shows on the x axis the simulated feature values,
      and on the y axis the uplift prediction under the assumption that the simulated
      feature takes the value given on the x axis. There are three curves on the
      graph: for low, middle and high confidence

    - a histogram graph of the feature simulated values

    :param ax: the axes where the uplift graph is plotted
    """

    _COLOR_CONFIDENCE = "silver"
    _COLOR_MEDIAN_UPLIFT = "orange"

    def __init__(self, ax: Optional[Axes] = None) -> None:
        super().__init__(ax=ax)
        self._ax_histogram: Optional[Axes] = None

    def drawing_start(self, title: str) -> None:
        self._draw_title(title)

    @property
    def ax_uplift_graph(self) -> Axes:
        return self.ax

    @property
    def ax_histogram(self) -> Optional[Axes]:
        return self._ax_histogram

    def _draw_title(self, title: str) -> None:
        """Print the title."""
        self.ax.figure.suptitle(title)

    def draw_uplift_graph(
        self,
        feature_name: str,
        target_name: str,
        partitioning: Partitioning,
        median_uplift: ListLike[T_Number],
        low_percentile_uplift: ListLike[T_Number],
        high_percentile_uplift: ListLike[T_Number],
        low_percentile: int,
        high_percentile: int,
    ):
        """
        Draw the graph with the uplift curves: median, low and high percentiles.
        """
        line_low = self._draw_low_confidence_uplift(
            partitioning.partitions(), low_percentile_uplift
        )
        line_median = self._draw_middle_confidence_uplift(
            partitioning.partitions(), median_uplift
        )
        line_high = self._draw_high_confidence_uplift(
            partitioning.partitions(), high_percentile_uplift
        )

        labels = [
            f"{high_percentile}th percentile",
            "Median",
            f"{low_percentile}th percentile",
        ]
        handles = [line_high, line_median, line_low]
        self._draw_uplift_legend(handles=handles, labels=labels)

        self._draw_uplift_axis(
            x_label=feature_name, y_label=f"Predicted mean uplift ({target_name})"
        )

        self._draw_null_uplift_line()
        self._set_spins()

    def draw_histogram(self, partitioning: Partitioning) -> None:
        """
        Draw frequencies histogram.

        :param partitioning: the partitioning used for the simulation
        """
        self._ax_histogram = self._make_ax_histogram()

        self._ax_histogram.bar(
            x=(partitioning.partitions()),
            height=(partitioning.frequencies()),
            color="silver",
            edgecolor="white",
        )
        self._ax_histogram.invert_yaxis()
        self._ax_histogram.tick_params(axis="y", labelcolor="black")
        max_y = max(partitioning.frequencies())
        label_vertical_offset = max_y * 0.05
        for (_x, _y) in zip(partitioning.partitions(), partitioning.frequencies()):
            if _y > 0:
                self._ax_histogram.text(
                    x=_x,
                    y=_y + label_vertical_offset,
                    s=str(_y),
                    horizontalalignment="center",
                    verticalalignment="top",
                )
        self._ax_histogram.get_yaxis().set_visible(False)
        self._ax_histogram.get_xaxis().set_visible(False)
        for pos in ["top", "right", "left", "bottom"]:
            self._ax_histogram.spines[pos].set_visible(False)

    def _make_ax_histogram(self) -> Axes:
        """
        Return the axes for the histogram.

        :return: the histogram axes
        """
        divider = make_axes_locatable(self.ax)
        histogram_axes = divider.append_axes(
            # todo: what is the meaning of Scaled in this context?
            position="bottom",
            size=Scaled(0.05),
            pad=Scaled(0.01),
        )
        return histogram_axes

    def _draw_extreme_uplift(
        self, x: Iterable[T_Number], y: Iterable[T_Number]
    ) -> Line2D:
        # draw curve of uplift for confidence interval
        line, = self.ax.plot(x, y, color=self._COLOR_CONFIDENCE)
        return line

    def _draw_low_confidence_uplift(
        self, x: Iterable[T_Number], y: Iterable[T_Number]
    ) -> Line2D:
        """Draw the low confidence uplift curve.

        :param x: simulated feature values
        :param y: low confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self._draw_extreme_uplift(x, y)

    def _draw_high_confidence_uplift(
        self, x: Iterable[T_Number], y: Iterable[T_Number]
    ) -> Line2D:
        """
        Draw the high confidence uplift curve.

        :param x: simulated feature values
        :param y: high confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self._draw_extreme_uplift(x, y)

    def _draw_middle_confidence_uplift(
        self, x: Iterable[T_Number], y: Iterable[T_Number]
    ) -> Line2D:
        """
        Draw the middle uplift curve.

        :param x: simulated feature values
        :param y: middle confidence simulated predictions
        :return: the matplotlib object representing the line plotted
        """
        return self.ax.plot(x, y, color=self._COLOR_MEDIAN_UPLIFT)

    def _draw_uplift_axis(self, x_label: str, y_label: str) -> None:
        """
        Draw x ticks, set x labels in the uplift graph.

        :param x_label: label for the x axis in the uplift graph
        :param y_label: label for the y axis in the uplift graph
        """

        self.ax.tick_params(axis="x", labelbottom=True, bottom=False)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

    def _draw_null_uplift_line(self) -> None:
        """Draw  a horizontal line y=0 on the uplift graph."""
        self.ax.axhline(y=0, linewidth=0.5)

    def _set_spins(self) -> None:
        """Set the spines (data boundary lines) in the uplift graph."""

        for pos in ["top", "right", "bottom"]:
            self.ax.spines[pos].set_visible(False)

    def _draw_uplift_legend(
        self, handles: Iterable[Artist], labels: Iterable[str]
    ) -> None:
        """
        Draw the legend of the uplift graph.

        :param handles: the matplotlib objects to include in the legend
        :param labels: the label in the legend for the matplotlib objects in ``handles``
        """

        self.ax.legend(handles, labels, frameon=False)
