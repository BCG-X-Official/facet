"""
Simulation drawer.

:class:`SimulationDrawer` draws a simulation plot with on the x axis the feature
values in the simulation and on the y axis the associated prediction uplift. Below
this graph there is a histogram of the feature values.
"""

from typing import Optional, TypeVar

from gamma.viz import ChartDrawer
from gamma.viz.simulation._style import SimulationMatplotStyle
from gamma.yieldengine.partition import RangePartitioning
from gamma.yieldengine.simulation import UnivariateSimulation

T_RangePartitioning = TypeVar("T_RangePartitioning", bound=RangePartitioning)


class SimulationDrawer(ChartDrawer[UnivariateSimulation, SimulationMatplotStyle]):
    """
    Simulation drawer with high/low confidence intervals.

    :param style: drawing style for the uplift graph and the feature histogram
    :param simulation: the data for the simulation
    :param title: title of the char
    :param histogram: if ``True`` (default) the feature histogram is plotted,
      if ``False`` the histogram is not plotted
    """

    def __init__(
        self,
        # simulation_data: SimulationData,
        style: SimulationMatplotStyle,
        simulation: UnivariateSimulation = None,
        title: Optional[str] = None,
        histogram: bool = True,
    ):
        super().__init__(title=title, model=simulation, style=style)
        self._style = style
        self._simulation = simulation
        self._title = title if title else "Simulation"
        self._histogram = histogram

    def _draw(self) -> None:
        # draw the simulation chart
        # self._style.initialize_chart(histogram=self._histogram)
        self._style.drawing_start(self._title)
        self._draw_uplift_graph()

        if self._histogram:
            self._draw_histogram()

        # self._draw_title()

    # def _draw_title(self) -> None:
    #     self._style.draw_title(self._title)

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
        return None

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
        return None
