"""
Simulation drawer.

:class:`SimulationDrawer` draws a simulation plot with on the x axis the feature
values in the simulation and on the y axis the associated prediction uplift. Below
this graph there is a histogram of the feature values.
"""

from typing import Optional

from gamma.viz import ChartDrawer
from gamma.viz.simulation._simulation_data import SimulationData, T_RangePartitioning
from gamma.viz.simulation._style import SimulationMatplotStyle
from gamma.yieldengine.partition import NumericType
from gamma.yieldengine.simulation import UnivariateSimulation


class SimulationDrawer(ChartDrawer[SimulationData, SimulationMatplotStyle]):
    """
    Simulation drawer with high/low confidence intervals.

    :param style: drawing style for the uplift graph and the feature histogram
    :param title: title of the char
    :param histogram: if ``True`` (default) the feature histogram is plotted,
      if ``False`` the histogram is not plotted
    """

    def __init__(
        self,
        # simulation_data: SimulationData,
        style: SimulationMatplotStyle,
        title: Optional[str] = None,
        histogram: bool = True,
        simulation: Optional[UnivariateSimulation] = None,
    ):
        super().__init__(title=title, model=simulation, style=style)
        # self._simulation_data = simulation
        self._style = style
        self._title = title if title else "Simulation"
        self._histogram = histogram
        self._simulation = simulation

    def _draw(self) -> None:
        # draw the simulation chart
        self._style.initialize_chart(histogram=self._histogram)

        self._draw_uplift_graph()

        if self._histogram:
            self._draw_histogram()

        self._draw_title()

    def _draw_title(self) -> None:
        self._style.draw_title(self._title)

    def _draw_uplift_graph(self) -> None:
        # draw the graph with the uplift curves
        feature = self._simulation.feature
        target_name = self._simulation.target_name
        feature_values = self._simulation.feature_values
        median_uplift = self._simulation.median_uplift
        low_percentile_uplift = self._simulation.low_percentile_uplift
        high_percentile_uplift = self._simulation.high_percentile_uplift
        low_percentile = self._simulation.low_percentile
        high_percentile = self._simulation.high_percentile

        self._style.draw_uplift_graph(
            feature=feature,
            target_name=target_name,
            feature_values=feature_values,
            median_uplift=median_uplift,
            low_percentile_uplift=low_percentile_uplift,
            high_percentile_uplift=high_percentile_uplift,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )

    def _draw_histogram(self) -> None:
        # draw the histogram of the simulation values

        feature_values = self._simulation.feature_values
        feature_frequencies = self._simulation.feature_frequencies
        partition_width = self._simulation.partition_width

        self._style.draw_histogram(
            feature_values=feature_values,
            feature_frequencies=feature_frequencies,
            partition_width=partition_width,
        )
        # self._style.ax = ax
        #
        # x = self._simulation_data.partitioning.partitions()
        # height = self._simulation_data.partitioning.frequencies()
        # partition_width = self._simulation_data.partitioning.partition_width
        #
        # self._style.draw_histogram(x=x, height=height, width=partition_width)
        return None

    def set_feature(self, feature: str) -> None:
        """
        Change the feature on which to simulate.

        :param feature: the new feature to use for the simulation
        """
        self._simulation.feature = feature
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
        self._simulation.update(
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
