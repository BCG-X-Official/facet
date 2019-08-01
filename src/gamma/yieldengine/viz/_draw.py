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
Simulation drawer.

:class:`SimulationDrawer` draws a simulation plot with on the x axis the feature
values in the simulation and on the y axis the associated prediction uplift. Below
this graph there is a histogram of the feature values.
"""

from typing import TypeVar

from gamma.viz import ChartDrawer
from gamma.yieldengine.partition import RangePartitioning
from gamma.yieldengine.simulation import UnivariateSimulation
from gamma.yieldengine.viz._style import SimulationStyle

T_RangePartitioning = TypeVar("T_RangePartitioning", bound=RangePartitioning)


class SimulationDrawer(ChartDrawer[UnivariateSimulation, SimulationStyle]):
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
        title: str,
        simulation: UnivariateSimulation,
        style: SimulationStyle,
        histogram: bool = True,
    ):
        super().__init__(title=title, model=simulation, style=style)
        self._histogram = histogram

    def _draw(self) -> None:
        # draw the simulation chart
        self._draw_uplift_graph()

        if self._histogram:
            self._draw_histogram()

    def _draw_uplift_graph(self) -> None:
        # draw the graph with the uplift curves
        simulation: UnivariateSimulation = self._model
        self._style.draw_uplift(
            feature_name=simulation.feature_name,
            target_name=simulation.target_name,
            partitioning=simulation.partitioning,
            median_uplift=simulation.median_uplift,
            min_uplift=simulation.min_uplift,
            max_uplift=simulation.max_uplift,
            min_percentile=simulation.min_percentile,
            max_percentile=simulation.max_percentile,
        )
        return None

    def _draw_histogram(self) -> None:
        # draw the histogram of the simulation values

        simulation: UnivariateSimulation = self._model
        self._style.draw_histogram(partitioning=simulation.partitioning)
        return None
