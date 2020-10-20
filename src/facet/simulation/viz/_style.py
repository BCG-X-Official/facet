"""
Drawing styles for simulation results.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, Tuple, TypeVar

from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from mpl_toolkits.axes_grid1.axes_size import Scaled

from pytools.api import AllTracker, inheritdoc
from pytools.viz import DrawStyle, MatplotStyle, TextStyle
from pytools.viz.colors import (
    RGBA_DARK_BLUE,
    RGBA_GREY,
    RGBA_LIGHT_BLUE,
    RGBA_LIGHT_GREEN,
)
from pytools.viz.text import format_table

log = logging.getLogger(__name__)

__all__ = ["SimulationStyle", "SimulationMatplotStyle", "SimulationReportStyle"]

#
# Type variables
#


T_Value = TypeVar("T_Value")
T_Number = TypeVar("T_Number", int, float)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class SimulationStyle(DrawStyle, metaclass=ABCMeta):
    """
    Base class of styles used by :class:`.SimulationDrawer`.
    """

    @abstractmethod
    def draw_uplift(
        self,
        feature: str,
        target: str,
        values_label: str,
        values_median: Sequence[T_Number],
        values_min: Sequence[T_Number],
        values_max: Sequence[T_Number],
        values_baseline: T_Number,
        percentile_lower: float,
        percentile_upper: float,
        partitions: Sequence[Any],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw the graph with the uplift curves: median, low and high percentiles.

        :param feature: name of the simulated feature
        :param target: name of the target for which output values were simulated
        :param values_label: label of the values axis
        :param values_median: median uplift values
        :param values_min: low percentile uplift values
        :param values_max: high percentile uplift values
        :param values_baseline: the baseline of the simulationb
        :param percentile_lower: percentile used to compute values_min
        :param percentile_upper: percentile used to compute values_max
        :param partitions: the partitioning (center values) of the simulated feature
        :param frequencies: observed frequencies for each partition
        :param is_categorical_feature: ``True`` if the simulated feature is categorical
        """
        pass

    @abstractmethod
    def draw_histogram(
        self,
        partitions: Sequence[Any],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw the histogram of observed value counts per partition.

        :param partitions: the partitioning (center values) of the simulated feature
        :param frequencies: observed frequencies for each partition
        :param is_categorical_feature: ``True`` if the simulated feature is \
            categorical, ``False`` otherwise
        """
        pass

    @staticmethod
    def _legend(percentile_lower: float, percentile_upper: float) -> Tuple[str, ...]:
        # generate a triple with legend names for the min percentile, median, and max
        # percentile
        return (
            f"{percentile_lower}th percentile",
            "Median",
            f"{percentile_upper}th percentile",
            "Baseline",
        )


@inheritdoc(match="[see superclass]")
class SimulationMatplotStyle(MatplotStyle, SimulationStyle):
    """
    Matplotlib Style for simulation chart.

    Along the range of simulated feature values on the x axis, plots the median and
    confidence intervals of the simulated target value.

    A bar chart below the plot shows a histogram of actually observed values near the
    simulated values.
    """

    _COLOR_CONFIDENCE_INTERVAL = RGBA_DARK_BLUE
    _COLOR_MEDIAN = RGBA_LIGHT_BLUE
    _COLOR_BASELINE = RGBA_LIGHT_GREEN
    _COLOR_BARS = RGBA_GREY
    _WIDTH_BARS = 0.8

    _HISTOGRAM_SIZE_RATIO = 1 / 3

    def draw_uplift(
        self,
        feature: str,
        target: str,
        values_label: str,
        values_median: Sequence[T_Number],
        values_min: Sequence[T_Number],
        values_max: Sequence[T_Number],
        values_baseline: T_Number,
        percentile_lower: float,
        percentile_upper: float,
        partitions: Sequence[Any],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """[see superclass]"""

        # draw the mean predicted uplift, showing median and confidence ranges for
        # each prediction
        if is_categorical_feature:
            x = range(len(partitions))
        else:
            x = partitions
        ax = self.ax
        (line_min,) = ax.plot(x, values_min, color=self._COLOR_CONFIDENCE_INTERVAL)
        (line_median,) = ax.plot(x, values_median, color=self._COLOR_MEDIAN)
        (line_max,) = ax.plot(x, values_max, color=self._COLOR_CONFIDENCE_INTERVAL)
        # add a horizontal line at the baseline
        line_base = ax.axhline(
            y=values_baseline, linewidth=0.5, color=self._COLOR_CONFIDENCE_BASELINE
        )

        # add a legend
        labels = self._legend(
            percentile_lower=percentile_lower, percentile_upper=percentile_upper
        )
        handles = (line_max, line_median, line_min, line_base)
        ax.legend(handles, labels)

        # label the y axis
        ax.set_ylabel(values_label)

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

        # remove the top and right spines
        for pos in ["top", "right"]:
            ax.spines[pos].set_visible(False)

    def draw_histogram(
        self,
        partitions: Sequence[T_Value],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """[see superclass]"""

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
                    self.renderer
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
                    s=str(y),
                    horizontalalignment="center",
                    verticalalignment="top",
                )

        # hide x and y axis
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # hide the spines
        for pos in ["top", "right", "left", "bottom"]:
            ax.spines[pos].set_visible(False)


@inheritdoc(match="[see superclass]")
class SimulationReportStyle(SimulationStyle, TextStyle):
    """
    Renders simulation results as a text report.
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
    _PARTITION_NUMBER_FORMAT = "g"

    # format for frequencies
    _FREQUENCY_WIDTH = 6
    _FREQUENCY_FORMAT = f"{_FREQUENCY_WIDTH}g"

    @staticmethod
    def _num_format(heading: str):
        return f"> {len(heading)}.{SimulationReportStyle._NUM_PRECISION}g"

    def _drawing_start(self, title: str) -> None:
        # print the report title
        self.out.write(f"{title}\n")

    def draw_uplift(
        self,
        feature: str,
        target: str,
        values_label: str,
        values_median: Sequence[T_Number],
        values_min: Sequence[T_Number],
        values_max: Sequence[T_Number],
        values_baseline: T_Number,
        percentile_lower: float,
        percentile_upper: float,
        partitions: Sequence[Any],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """[see superclass]"""

        out = self.out
        self.out.write(f"\n{values_label}:\n\nBaseline = {values_baseline}\n\n")
        out.write(
            format_table(
                headings=[
                    self._PARTITION_HEADING,
                    *self._legend(
                        percentile_lower=percentile_lower,
                        percentile_upper=percentile_upper,
                    )[:3],
                ],
                formats=[
                    self._partition_format(is_categorical_feature),
                    *([self._NUM_FORMAT] * 3),
                ],
                data=list(zip(partitions, values_min, values_median, values_max)),
                alignment=["<", ">", ">", ">"],
            )
        )

    def draw_histogram(
        self,
        partitions: Sequence[T_Value],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """[see superclass]"""

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
        # print two trailing line breaks
        self.out.write("\n")

    def _partition_format(self, is_categorical: bool) -> str:
        if is_categorical:
            return self._PARTITION_TEXT_FORMAT
        else:
            return self._PARTITION_NUMBER_FORMAT


__tracker.validate()
