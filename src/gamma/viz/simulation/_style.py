"""
Simulation drawing styles

:class:`SimulationMatplotStyle` draws some simulated low, middle and high prediction
uplift.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from mpl_toolkits.axes_grid1.axes_size import Scaled

from gamma import ListLike
from gamma.viz import ChartStyle, MatplotStyle
from gamma.yieldengine.partition import Partitioning, T_Number

log = logging.getLogger(__name__)


class SimulationStyle(ChartStyle, ABC):
    @abstractmethod
    def draw_uplift(
        self,
        feature_name: str,
        target_name: str,
        partitioning: Partitioning,
        median_uplift: ListLike[T_Number],
        min_uplift: ListLike[T_Number],
        max_uplift: ListLike[T_Number],
        low_percentile: int,
        high_percentile: int,
    ) -> None:
        """
        Draw the graph with the uplift curves: median, low and high percentiles.
        """
        pass

    @abstractmethod
    def draw_histogram(self, partitioning: Partitioning) -> None:
        """
        Draw frequencies histogram.

        :param partitioning: the partitioning used for the simulation
        """
        pass


class SimulationMatplotStyle(MatplotStyle, SimulationStyle):
    """
    Matplotlib Style for simulation chart.

    Allows to plot two different graph:

    - an uplift graph that shows on the x axis the simulated feature values,
      and on the y axis the uplift prediction under the assumption that the simulated
      feature takes the value given on the x axis. There are three curves on the
      graph: for low, middle and high confidence

    - a histogram graph of the feature simulated values

    :param ax: the axes where the uplift graph is plotted
    :param x_label_height: the height of the central chart area reserved for the x
      labels, relative to the overall chart height (default: 0.1)
    """

    _COLOR_CONFIDENCE = "silver"
    _COLOR_BARS = "silver"
    _COLOR_MEDIAN_UPLIFT = "orange"

    def __init__(self, ax: Optional[Axes] = None, x_label_height: float = 0.1) -> None:
        super().__init__(ax=ax)
        if x_label_height <= 0 or x_label_height >= 1:
            raise ValueError(f"arg x_label_height={x_label_height} must be > 0 and < 1")
        self._x_label_height = x_label_height

    def draw_uplift(
        self,
        feature_name: str,
        target_name: str,
        partitioning: Partitioning,
        median_uplift: ListLike[T_Number],
        min_uplift: ListLike[T_Number],
        max_uplift: ListLike[T_Number],
        low_percentile: int,
        high_percentile: int,
    ) -> None:
        """
        Draw the graph with the uplift curves: median, low and high percentiles.
        """

        x = self._x_labels(partitioning)
        ax = self.ax
        line_min, = ax.plot(x, min_uplift, color=self._COLOR_CONFIDENCE)
        line_median, = ax.plot(x, median_uplift, color=self._COLOR_MEDIAN_UPLIFT)
        line_max, = ax.plot(x, max_uplift, color=self._COLOR_CONFIDENCE)

        labels = [
            f"{high_percentile}th percentile",
            "Median",
            f"{low_percentile}th percentile",
        ]
        handles = [line_max, line_median, line_min]
        ax.legend(handles, labels)

        ax.tick_params(axis="x", labelbottom=True, bottom=False)
        ax.set_ylabel(f"Predicted mean uplift ({target_name})")

        if partitioning.is_categorical:
            ax.set_xticks(x)
            ax.set_xticklabels(labels=partitioning.partitions(), rotation=45)

        ax.axhline(y=0, linewidth=0.5)
        for pos in ["top", "right", "bottom"]:
            ax.spines[pos].set_visible(False)

    def draw_histogram(self, partitioning: Partitioning) -> None:
        """
        Draw frequencies histogram.

        :param partitioning: the partitioning used for the simulation
        """
        x_values = list(range(len(partitioning)))
        y_values = partitioning.frequencies()

        # create the sub-axes for the histogram
        uplift_y_min, uplift_y_max = self.ax.get_ylim()
        uplift_height = abs(uplift_y_max - uplift_y_min)

        divider: AxesDivider = make_axes_locatable(self.ax)
        ax = divider.append_axes(
            position="bottom",
            size=Scaled(uplift_height),
            pad=Scaled(
                2 * self._x_label_height * uplift_height / (1 - self._x_label_height)
            ),
        )

        ax.invert_yaxis()

        ax.bar(x=x_values, height=y_values, color=self._COLOR_BARS, align="center")

        ax.set_xmargin(0)

        # draw labels
        _, max_y = ax.get_ylim()
        label_vertical_offset = max_y * 0.02
        for x, y in zip(x_values, y_values):
            if y > 0:
                ax.text(
                    x=x,
                    y=y + label_vertical_offset,
                    s=y,
                    horizontalalignment="center",
                    verticalalignment="top",
                )

        # ax.set_xticks(x_values)

        # hide x and y axis
        # ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for pos in ["top", "right", "left", "bottom"]:
            ax.spines[pos].set_visible(False)

    def _x_labels(self, partitioning) -> ListLike:
        if partitioning.is_categorical:
            return list(range(len(partitioning)))
        else:
            return partitioning.partitions()
