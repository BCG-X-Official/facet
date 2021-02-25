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
from pytools.text import format_table
from pytools.viz import DrawingStyle, MatplotStyle, TextStyle

log = logging.getLogger(__name__)

__all__ = ["SimulationStyle", "SimulationMatplotStyle", "SimulationReportStyle"]

#
# Type variables
#


T_Partition = TypeVar("T_Partition")

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class SimulationStyle(DrawingStyle, metaclass=ABCMeta):
    """
    Base class of styles used by :class:`.SimulationDrawer`.
    """

    @abstractmethod
    def draw_uplift(
        self,
        feature_name: str,
        output_name: str,
        output_unit: str,
        outputs_median: Sequence[float],
        outputs_lower_bound: Sequence[float],
        outputs_upper_bound: Sequence[float],
        baseline: float,
        confidence_level: float,
        partitions: Sequence[Any],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """
        Draw the graph with the uplift curves: median, low and high percentiles.

        :param feature_name: name of the simulated feature
        :param output_name: name of the target for which output values were simulated
        :param output_unit: the unit of the output axis
        :param outputs_median: the medians of the simulated outputs
        :param outputs_lower_bound: the lower CI bounds of the simulated outputs
        :param outputs_upper_bound: the upper CI bounds of the simulated outputs
        :param baseline: the baseline of the simulation
        :param confidence_level: the confidence level used to calculate the CI bounds
        :param partitions: the central or categorical values representing the partitions
        :param frequencies: observed frequencies of the partitions
        :param is_categorical_feature: ``True`` if the simulated feature is
            categorical; ``False`` otherwise
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
        :param is_categorical_feature: ``True`` if the simulated feature is
            categorical, ``False`` otherwise
        """
        pass

    @staticmethod
    def _legend(confidence_level: float) -> Tuple[str, ...]:
        # generate a triple with legend names for the min percentile, median, and max
        # percentile
        tail_percentile = (100.0 - confidence_level * 100.0) / 2
        return (
            f"{tail_percentile}th percentile",
            "Median",
            f"{100.0 - tail_percentile}th percentile",
            "Baseline",
        )


@inheritdoc(match="[see superclass]")
class SimulationMatplotStyle(MatplotStyle, SimulationStyle):
    """
    `matplotlib` style for simulation chart.

    Along the range of simulated feature values on the x axis, plots the median and
    confidence intervals of the simulated target value.

    A bar chart below the plot shows a histogram of actually observed values near the
    simulated values.
    """

    # sizing constants
    __WIDTH_BARS = 0.8
    __HISTOGRAM_SIZE_RATIO = 1 / 3

    def draw_uplift(
        self,
        feature_name: str,
        output_name: str,
        output_unit: str,
        outputs_median: Sequence[float],
        outputs_lower_bound: Sequence[float],
        outputs_upper_bound: Sequence[float],
        baseline: float,
        confidence_level: float,
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

        # plot the confidence bounds and the median
        (line_min,) = ax.plot(x, outputs_lower_bound, color=self.colors.accent_3)
        (line_median,) = ax.plot(x, outputs_median, color=self.colors.accent_2)
        (line_max,) = ax.plot(x, outputs_upper_bound, color=self.colors.accent_3)

        # add a horizontal line at the baseline
        line_base = ax.axhline(y=baseline, linewidth=0.5, color=self.colors.accent_1)

        # add a legend
        labels = self._legend(confidence_level=confidence_level)
        handles = (line_max, line_median, line_min, line_base)
        ax.legend(handles, labels)

        # label the y axis
        ax.set_ylabel(output_unit)

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
        partitions: Sequence[T_Partition],
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
                size=Scaled(
                    uplift_height * SimulationMatplotStyle.__HISTOGRAM_SIZE_RATIO
                ),
                pad=Scaled(
                    axis_below_size_data
                    * (uplift_height / (uplift_height - axis_below_size_data))
                    * (1 + SimulationMatplotStyle.__HISTOGRAM_SIZE_RATIO)
                ),
            )

        ax = _make_sub_axes()

        self.apply_color_scheme(ax)

        ax.invert_yaxis()

        width_bars = SimulationMatplotStyle.__WIDTH_BARS

        # reduce the horizontal margin such that half a bar is to the left of the
        # leftmost tick mark (but the tick mark stays aligned with the main
        # simulation chart)
        x_margin, _ = ax.margins()
        ax.set_xmargin(
            max(
                0,
                (width_bars / 2 - x_margin * (n_partitions - 1))
                / (width_bars - (n_partitions - 1)),
            )
        )

        # draw the histogram bars
        ax.bar(
            x=x_values,
            height=y_values,
            color=self.colors.fill_1,
            align="center",
            width=width_bars,
        )

        # padding between bars and labels is 2% of histogram height; might want to
        # replace this with a measure based on text height
        label_vertical_offset = max(y_values) * 0.02

        # draw labels
        color_fg = self.colors.foreground
        for x, y in zip(x_values, y_values):
            if y > 0:
                ax.text(
                    x=x,
                    y=y + label_vertical_offset,
                    s=str(y),
                    horizontalalignment="center",
                    verticalalignment="top",
                    color=color_fg,
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
    __NUM_PRECISION = 3
    __NUM_WIDTH = __NUM_PRECISION + 6
    __NUM_FORMAT = f"< {__NUM_WIDTH}.{__NUM_PRECISION}g"

    # table headings
    __HEADING_PARTITION = "Partition"
    __HEADING_FREQUENCY = "Frequency"

    # format for partitions
    __PARTITION_TEXT_FORMAT = "s"
    __PARTITION_NUMBER_FORMAT = "g"

    # format for frequencies
    __FREQUENCY_WIDTH = 6
    __FREQUENCY_FORMAT = f"{__FREQUENCY_WIDTH}g"

    @staticmethod
    def _num_format(heading: str):
        return f"> {len(heading)}.{SimulationReportStyle.__NUM_PRECISION}g"

    def draw_uplift(
        self,
        feature_name: str,
        output_name: str,
        output_unit: str,
        outputs_median: Sequence[float],
        outputs_lower_bound: Sequence[float],
        outputs_upper_bound: Sequence[float],
        baseline: float,
        confidence_level: float,
        partitions: Sequence[Any],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """[see superclass]"""

        out = self.out
        self.out.write(f"\n{output_unit}:\n\nBaseline = {baseline}\n\n")
        out.write(
            format_table(
                headings=[
                    SimulationReportStyle.__HEADING_PARTITION,
                    *self._legend(confidence_level=confidence_level)[:3],
                ],
                formats=[
                    self._partition_format(is_categorical=is_categorical_feature),
                    *([SimulationReportStyle.__NUM_FORMAT] * 3),
                ],
                data=list(
                    zip(
                        partitions,
                        outputs_lower_bound,
                        outputs_median,
                        outputs_upper_bound,
                    )
                ),
                alignment=["<", ">", ">", ">"],
            )
        )

    def draw_histogram(
        self,
        partitions: Sequence[T_Partition],
        frequencies: Sequence[int],
        is_categorical_feature: bool,
    ) -> None:
        """[see superclass]"""

        self.out.write("\nObserved frequencies:\n\n")
        self.out.write(
            format_table(
                headings=(
                    SimulationReportStyle.__HEADING_PARTITION,
                    SimulationReportStyle.__HEADING_FREQUENCY,
                ),
                data=list(zip(partitions, frequencies)),
                formats=(
                    self._partition_format(is_categorical=is_categorical_feature),
                    SimulationReportStyle.__FREQUENCY_FORMAT,
                ),
                alignment=["<", ">"],
            )
        )

    def finalize_drawing(self, **kwargs: Any) -> None:
        """[see superclass]"""
        super().finalize_drawing(**kwargs)
        # print two trailing line breaks
        self.out.write("\n")

    @staticmethod
    def _partition_format(is_categorical: bool) -> str:
        if is_categorical:
            return SimulationReportStyle.__PARTITION_TEXT_FORMAT
        else:
            return SimulationReportStyle.__PARTITION_NUMBER_FORMAT


__tracker.validate()
