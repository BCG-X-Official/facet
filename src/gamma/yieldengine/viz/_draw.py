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

from typing import *

from gamma.common import ListLike
from gamma.viz import Drawer
from gamma.yieldengine.partition import T_Value
from gamma.yieldengine.simulation import UnivariateSimulation
from gamma.yieldengine.viz._style import SimulationStyle


class SimulationDrawer(Drawer[UnivariateSimulation, SimulationStyle]):
    """
    Simulation drawer with high/low confidence intervals.

    :param style: drawing style for the uplift graph and the feature histogram
    :param simulation: the data for the simulation
    :param title: title of the char
    :param histogram: if ``True`` (default) the feature histogram is plotted,
      if ``False`` the histogram is not plotted
    """

    class _SimulationSeries(NamedTuple):
        # A set of aligned series representing the simulation result
        median_uplift: ListLike[T_Value]
        min_uplift: ListLike[T_Value]
        max_uplift: ListLike[T_Value]
        partitions: ListLike[T_Value]
        frequencies: ListLike[T_Value]

    def __init__(
        self,
        simulation: UnivariateSimulation,
        style: SimulationStyle,
        title: Optional[str] = None,
        histogram: bool = True,
    ):
        super().__init__(
            model=simulation,
            style=style,
            title=simulation.feature_name if title is None else title,
        )
        self._histogram = histogram

    def _draw(self) -> None:
        # draw the simulation chart
        simulation_series = self._get_simulation_series()

        self._draw_uplift_graph(simulation_series)

        if self._histogram:
            self._draw_histogram(simulation_series)

    def _draw_uplift_graph(self, simulation_series: _SimulationSeries) -> None:
        # draw the graph with the uplift curves
        simulation: UnivariateSimulation = self._model

        self._style.draw_uplift(
            feature_name=simulation.feature_name,
            target_name=simulation.target_name,
            min_percentile=simulation.min_percentile,
            max_percentile=simulation.max_percentile,
            is_categorical_feature=simulation.partitioning.is_categorical,
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            median_uplift=simulation_series.median_uplift,
            min_uplift=simulation_series.min_uplift,
            max_uplift=simulation_series.max_uplift,
        )
        return None

    def _draw_histogram(self, simulation_series: _SimulationSeries) -> None:
        # draw the histogram of the simulation values

        self._style.draw_histogram(
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            is_categorical_feature=self._model.partitioning.is_categorical,
        )
        return None

    def _get_simulation_series(self) -> _SimulationSeries:
        # return the simulation series for median uplift, min uplift, max uplift,
        # partitions and frequencies
        # If the partitioning of the simulation is categorical, the series are
        # sorted in ascending order of the median uplift.
        # Otherwise, the simulation series are returned unchanged.

        simulation: UnivariateSimulation = self.model

        simulation_series = self._SimulationSeries(
            simulation.median_change,
            simulation.min_change,
            simulation.max_change,
            simulation.partitioning.partitions(),
            simulation.partitioning.frequencies(),
        )

        if simulation.partitioning.is_categorical:
            return self._SimulationSeries(
                *zip(*sorted(zip(*simulation_series), key=lambda x: x[0]))
            )
        else:
            return simulation_series
