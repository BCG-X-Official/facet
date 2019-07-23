"""
Simulation drawer.

:class:`SimulationDrawer` draws a simulation plot with on the x axis the feature
values in the simulation and on the y axis the associated prediction uplift. Below
this graph there is a histogram of the feature values.
"""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from gamma.viz import ChartDrawer
from gamma.viz.simulation._simulation_data import SimulationData, T_RangePartitioning
from gamma.viz.simulation._style import SimulationMatplotStyle
from gamma.yieldengine.partition import NumericType


class SimulationDrawer(ChartDrawer[SimulationData, SimulationMatplotStyle]):
    """
    Simulation drawer with high/low confidence intervals.

    :param simulation_data: data used for the simulation
    :param style: drawing style for the uplift graph and the feature histogram
    :param title: title of the char
    :param histogram: if ``True`` (default) the feature histogram is plotted,
      if ``False`` the histogram is not plotted
    """

    def __init__(
        self,
        simulation_data: SimulationData,
        style: SimulationMatplotStyle,
        title: Optional[str] = None,
        histogram: bool = True,
    ):
        super().__init__(title=title, model=simulation_data, style=style)
        self._simulation_data = simulation_data
        self._style = style
        self._title = title if title else "Simulation"
        self._histogram = histogram

    def _draw(self) -> None:
        # draw the simulation chart
        if self._histogram:
            fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10), sharex="all")
            self._draw_uplift_axes(ax1)
            self._draw_histogram_axes(ax2)
        else:
            fig, ax1 = plt.subplots(nrows=1, figsize=(10, 10))
            self._draw_uplift_axes(ax1)
        return None

    def _draw_uplift_axes(self, ax: Axes) -> None:
        # draw the uplift curves
        self._style.ax = ax

        line_low = self.style.draw_low_confidence_uplift(
            *self._simulation_data.low_confidence_uplift
        )

        line_middle = self.style.draw_middle_confidence_uplift(
            *self._simulation_data.middle_confidence_uplift
        )

        line_high = self.style.draw_high_confidence_uplift(
            *self._simulation_data.high_confidence_uplift
        )

        handles = [line_high, line_middle, line_low]
        labels = ["90th percentile", "Median", "10th percentile"]
        self._style.draw_uplift_legend(handles=handles, labels=labels)

        xaxis_label = self._simulation_data.feature
        x_ticks = self._simulation_data.partitioning.partitions()
        yaxis_label = f"Predicted mean uplift ({self._simulation_data.target_name})"
        self.style.draw_uplift_axis(
            xaxis_label=xaxis_label, yaxis_label=yaxis_label, x_ticks=x_ticks
        )

        self._style.draw_null_uplift_line()

        self._style.set_spins()

        self._style.draw_title(self._title)
        return None

    def _draw_histogram_axes(self, ax: Axes) -> None:
        # draw the histogram of the simulation values
        self._style.ax = ax

        x = self._simulation_data.partitioning.partitions()
        height = self._simulation_data.partitioning.frequencies()
        partition_width = self._simulation_data.partitioning.partition_width

        self._style.draw_histogram(x=x, height=height, width=partition_width)
        return None

    def set_feature(self, feature: str) -> None:
        """
        Change the feature on which to simulate.

        :param feature: the new feature to use for the simulation
        """
        self._simulation_data.feature = feature
        return None

    def update_simulation(
        self,
        feature: Optional[str] = None,
        max_partitions: Optional[int] = None,
        lower_bound: Optional[NumericType] = None,
        upper_bound: Optional[NumericType] = None,
        partition_type: Optional[T_RangePartitioning] = None,
    ):
        """
        Update the partition parameters and re-run the simulation.

        :param feature: new feature for the simulation
        :param max_partitions: maximum number of partitions for the simulation
        :param lower_bound: minimum value for the simulated feature values
        :param upper_bound: maximum value for the simulated feature values
        :param partition_type: class to use for the partitioning
        """
        self._simulation_data.update(
            feature=feature,
            max_partitions=max_partitions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            partition_type=partition_type,
        )

    @property
    def histogram(self) -> bool:
        """If ``True`` the simulation shows a histogram of the simulated feature."""
        return self._histogram

    @histogram.setter
    def histogram(self, value: bool):
        self._histogram = value
