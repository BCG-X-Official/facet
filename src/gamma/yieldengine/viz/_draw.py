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

    def __init__(self, style: SimulationStyle, histogram: bool = True):
        super().__init__(style=style)
        self._histogram = histogram

    def _draw(self, data: UnivariateSimulation) -> None:
        # draw the simulation chart
        simulation_series = self._get_simulation_series(simulation=data)

        # draw the graph with the uplift curves
        self._style.draw_uplift(
            feature_name=data.feature_name,
            target_name=data.target_name,
            min_percentile=data.min_percentile,
            max_percentile=data.max_percentile,
            is_categorical_feature=data.partitioning.is_categorical,
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            median_uplift=simulation_series.median_uplift,
            min_uplift=simulation_series.min_uplift,
            max_uplift=simulation_series.max_uplift,
        )

        if self._histogram:
            # draw the histogram of the simulation values
            self._style.draw_histogram(
                partitions=simulation_series.partitions,
                frequencies=simulation_series.frequencies,
                is_categorical_feature=data.partitioning.is_categorical,
            )

    def _get_simulation_series(
        self, simulation: UnivariateSimulation
    ) -> _SimulationSeries:
        # return the simulation series for median uplift, min uplift, max uplift,
        # partitions and frequencies
        # If the partitioning of the simulation is categorical, the series are
        # sorted in ascending order of the median uplift.
        # Otherwise, the simulation series are returned unchanged.

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
