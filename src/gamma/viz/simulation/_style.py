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
    """

    _COLOR_CONFIDENCE = "silver"
    _COLOR_BARS = "silver"
    _COLOR_MEDIAN_UPLIFT = "orange"
    _WIDTH_BARS = 0.8

    _HISTOGRAM_SIZE_RATIO = 1 / 3

    def __init__(self, ax: Optional[Axes] = None) -> None:
        super().__init__(ax=ax)

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

        if partitioning.is_categorical:
            x = list(range(len(partitioning)))
        else:
            x = partitioning.partitions()
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

        ax.tick_params(
            axis="x",
            labelbottom=True,
            bottom=True,
            labelrotation=45 if partitioning.is_categorical else 0,
        )
        ax.set_ylabel(f"Mean predicted uplift ({target_name})")

        if partitioning.is_categorical or True:
            ax.set_xticks(x)
            ax.set_xticklabels(labels=partitioning.partitions())

        ax.axhline(y=0, linewidth=0.5)
        for pos in ["top", "right", "bottom"]:
            ax.spines[pos].set_visible(False)

    def draw_histogram(self, partitioning: Partitioning) -> None:
        """
        Draw frequencies histogram.

        :param partitioning: the partitioning used for the simulation
        """

        # get histogram size and values (horizontally, we count bars from 0..n-1
        n_partitions = len(partitioning)
        x_values = list(range(n_partitions))
        y_values = partitioning.frequencies()

        def _make_sub_axes() -> Axes:
            # create the sub-axes for the histogram

            parent_ax = self.ax
            y_min, y_max = parent_ax.get_ylim()
            uplift_height = abs(y_max - y_min)

            _, text_height = self.text_extent("A")

            vert_size = text_height + max(
                h
                for _, h in (
                    self.text_extent(
                        label, rotation=45 if partitioning.is_categorical else 0
                    )
                    for label in partitioning.partitions()
                )
            )

            divider: AxesDivider = make_axes_locatable(parent_ax)
            return divider.append_axes(
                position="bottom",
                size=Scaled(uplift_height * self._HISTOGRAM_SIZE_RATIO),
                pad=Scaled(
                    vert_size
                    * (uplift_height / (uplift_height - vert_size))
                    * (1 + self._HISTOGRAM_SIZE_RATIO)
                ),
            )

        ax = _make_sub_axes()

        ax.invert_yaxis()

        # reduce the margin such that half a bar is to the left of the leftmost tickmark
        x_margin, _ = ax.margins()
        ax.set_xmargin(
            max(
                0,
                (self._WIDTH_BARS / 2 - x_margin * (n_partitions - 1))
                / (self._WIDTH_BARS - (n_partitions - 1)),
            )
        )

        # draw the histogram bars
        ax.bar(
            x=x_values,
            height=y_values,
            color=self._COLOR_BARS,
            align="center",
            width=self._WIDTH_BARS,
        )

        # draw labels
        min_y, max_y = ax.get_ylim()
        label_vertical_offset = abs(max_y - min_y) * 0.05
        for x, y in zip(x_values, y_values):
            if y > 0:
                ax.text(
                    x=x,
                    y=y + label_vertical_offset,
                    s=y,
                    horizontalalignment="center",
                    verticalalignment="top",
                )

        # hide x and y axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # hide the spines
        for pos in ["top", "right", "left", "bottom"]:
            ax.spines[pos].set_visible(False)
