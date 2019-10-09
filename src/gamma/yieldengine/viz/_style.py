#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Simulation drawing styles

:class:`SimulationMatplotStyle` draws some simulated low, middle and high prediction
uplift.
"""

import logging
from abc import ABC, abstractmethod
from typing import *

from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from mpl_toolkits.axes_grid1.axes_size import Scaled

from gamma.common import ListLike
from gamma.viz import DrawStyle, MatplotStyle, TextStyle
from gamma.viz.text import format_table
from gamma.yieldengine.partition import T_Number, T_Value

log = logging.getLogger(__name__)


class SimulationStyle(DrawStyle, ABC):
    """
    The abstract simulation style known to the simulation drawer.
    """

    @abstractmethod
    def draw_uplift(
        self,
        feature: str,
        target: str,
        median_uplift: ListLike[T_Number],
        min_uplift: ListLike[T_Number],
        max_uplift: ListLike[T_Number],
        min_percentile: float,
        max_percentile: float,
        partitions: ListLike[Any],
        frequencies: ListLike[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw the graph with the uplift curves: median, low and high percentiles.
        """
        pass

    @abstractmethod
    def draw_histogram(
        self,
        partitions: ListLike[Any],
        frequencies: ListLike[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw frequencies histogram.
        """
        pass

    @staticmethod
    def _uplift_label(target_name: str) -> str:
        return f"Mean predicted uplift ({target_name})"

    @staticmethod
    def _legend(min_percentile: float, max_percentile: float) -> Tuple[str, str, str]:
        # generate a triple with legend names for the min percentile, median, and max
        # percentile
        return (
            f"{min_percentile}th percentile",
            "median",
            f"{max_percentile}th " f"percentile",
        )


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

    _COLOR_CONFIDENCE = "blue"
    _COLOR_BARS = "silver"
    _COLOR_MEDIAN_UPLIFT = "orange"
    _WIDTH_BARS = 0.8

    _HISTOGRAM_SIZE_RATIO = 1 / 3

    def __init__(self, ax: Optional[Axes] = None) -> None:
        super().__init__(ax=ax)

    def draw_uplift(
        self,
        feature: str,
        target: str,
        median_uplift: ListLike[T_Number],
        min_uplift: ListLike[T_Number],
        max_uplift: ListLike[T_Number],
        min_percentile: float,
        max_percentile: float,
        partitions: ListLike[Any],
        frequencies: ListLike[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw the uplift graph.

        :param feature: name of the simulated feature
        :param target: name of the target
        :param median_uplift: median uplift values
        :param min_uplift: low percentile uplift values
        :param max_uplift: high percentile uplift values
        :param min_percentile: percentile used to compute min_uplift
        :param max_percentile: percentile used to compute max_uplift
        :param partitions: partition (center) values
        :param frequencies: frequencies corresponding to the partitions
        :param is_categorical_feature: indicator of a categorical feature
        """

        # draw the mean predicted uplift, showing median and confidence ranges for
        # each prediction
        if is_categorical_feature:
            # x = list(range(len(partitioning)))
            x = range(len(partitions))
        else:
            # x = partitioning.partitions()
            x = partitions
        ax = self.ax
        line_min, = ax.plot(x, min_uplift, color=self._COLOR_CONFIDENCE)
        line_median, = ax.plot(x, median_uplift, color=self._COLOR_MEDIAN_UPLIFT)
        line_max, = ax.plot(x, max_uplift, color=self._COLOR_CONFIDENCE)

        # add a legend
        labels = self._legend(
            min_percentile=min_percentile, max_percentile=max_percentile
        )
        handles = [line_max, line_median, line_min]
        ax.legend(handles, labels)

        # label the y axis
        ax.set_ylabel(self._uplift_label(target_name=target))

        # format and label the x axis
        ax.tick_params(
            axis="x",
            labelbottom=True,
            bottom=True,
            labelrotation=45 if is_categorical_feature else 0,
        )
        if is_categorical_feature or True:
            ax.set_xticks(x)
            ax.set_xticklabels(labels=partitions)

        # add a horizontal line at y=0
        ax.axhline(y=0, linewidth=0.5)

        # remove the top and right spines
        for pos in ["top", "right"]:
            ax.spines[pos].set_visible(False)

    def draw_histogram(
        self,
        partitions: ListLike[T_Value],
        frequencies: ListLike[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw frequencies histogram.
        """

        # get histogram size and values (horizontally, we count bars from 0..n-1
        n_partitions = len(partitions)
        x_values = list(range(n_partitions))
        y_values = frequencies

        def _make_sub_axes() -> Axes:
            # create the sub-axes for the histogram

            # get the height of the main axes - this will be the basis for
            # calculating the size of the new sub-axes for the histogram
            main_ax = self.ax
            y_min, y_max = main_ax.get_ylim()
            uplift_height = abs(y_max - y_min)

            def _x_axis_height() -> float:
                _, axis_below_size_pixels = main_ax.get_xaxis().get_text_heights(
                    self._renderer
                )
                ((_, y0), (_, y1)) = main_ax.transData.inverted().transform(
                    ((0, 0), (0, axis_below_size_pixels))
                )
                return abs(y1 - y0)

            # calculate the height of the x axis in data space; add additional padding
            axis_below_size_data = _x_axis_height() * 1.2

            # create the axes divider, then use it to append the new sub-axes at the
            # bottom while leaving sufficient padding in-between to accommodate the
            # main axes' x axis labels
            divider: AxesDivider = make_axes_locatable(main_ax)
            return divider.append_axes(
                position="bottom",
                size=Scaled(uplift_height * self._HISTOGRAM_SIZE_RATIO),
                pad=Scaled(
                    axis_below_size_data
                    * (uplift_height / (uplift_height - axis_below_size_data))
                    * (1 + self._HISTOGRAM_SIZE_RATIO)
                ),
            )

        ax = _make_sub_axes()

        ax.invert_yaxis()

        # reduce the horizontal margin such that half a bar is to the left of the
        # leftmost tickmark (but the tickmark stays aligned with the main
        # simulation chart)
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

        # padding between bars and labels is 2% of histogram height; might want to
        # replace this with a measure based on text height
        label_vertical_offset = max(y_values) * 0.02

        # draw labels
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


class SimulationReportStyle(SimulationStyle, TextStyle):
    """
    Simulation results as a text report
    """

    # general format wih sufficient space for potential sign and "e" notation
    _NUM_PRECISION = 3
    _NUM_WIDTH = _NUM_PRECISION + 6
    _NUM_FORMAT = f"< {_NUM_WIDTH}.{_NUM_PRECISION}g"

    # table headings
    _PARTITION_HEADING = "Partition"
    _FREQUENCY_HEADING = "Frequency"

    # format for partitions
    _PARTITION_TEXT_FORMAT = "s"
    _PARTITION_NUMBER_FORMAT = ".3g"

    # format for frequencies
    _FREQUENCY_WIDTH = 6
    _FREQUENCY_FORMAT = f"{_FREQUENCY_WIDTH}g"

    @staticmethod
    def _num_format(heading: str):
        return f"> {len(heading)}.{SimulationReportStyle._NUM_PRECISION}g"

    def _drawing_start(self, title: str) -> None:
        """
        Print the report title.
        """
        self.out.write(f"SIMULATION REPORT: {title}\n")

    def draw_uplift(
        self,
        feature: str,
        target: str,
        median_uplift: ListLike[T_Number],
        min_uplift: ListLike[T_Number],
        max_uplift: ListLike[T_Number],
        min_percentile: float,
        max_percentile: float,
        partitions: ListLike[Any],
        frequencies: ListLike[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Print the uplift report.
        """
        out = self.out
        self.out.write(f"\n{self._uplift_label(target_name=target)}:\n\n")
        out.write(
            format_table(
                headings=[
                    self._PARTITION_HEADING,
                    *self._legend(
                        min_percentile=min_percentile, max_percentile=max_percentile
                    ),
                ],
                formats=[
                    self._partition_format(is_categorical_feature),
                    *([self._NUM_FORMAT] * 3),
                ],
                data=list(zip(partitions, min_uplift, median_uplift, max_uplift)),
                alignment=["<", ">", ">", ">"],
            )
        )

    def draw_histogram(
        self,
        partitions: ListLike[T_Value],
        frequencies: ListLike[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Print the histogram report.
        :param is_categorical_feature:
        :param partitions:
        :param frequencies:
        """
        self.out.write("\nObserved frequencies:\n\n")
        self.out.write(
            format_table(
                headings=(self._PARTITION_HEADING, self._FREQUENCY_HEADING),
                data=list(zip(partitions, frequencies)),
                formats=(
                    self._partition_format(is_categorical=is_categorical_feature),
                    self._FREQUENCY_FORMAT,
                ),
                alignment=["<", ">"],
            )
        )

    def _drawing_finalize(self) -> None:
        """
        Print two trailing line breaks.
        """
        self.out.write("\n")

    def _partition_format(self, is_categorical: bool) -> str:
        if is_categorical:
            return self._PARTITION_TEXT_FORMAT
        else:
            return self._PARTITION_NUMBER_FORMAT
