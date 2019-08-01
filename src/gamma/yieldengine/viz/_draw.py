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

from typing import NamedTuple, TypeVar

from gamma import ListLike
from gamma.viz import ChartDrawer
from gamma.yieldengine.partition import RangePartitioning, T_Value
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

    F_MEDIAN_UPLIFT = "median_uplift"
    F_MIN_UPLIFT = "min_uplift"
    F_MAX_UPLIFT = "max_uplift"
    F_FREQUENCIES = "frequencies"
    F_PARTITIONS = "partitions"

    class SimulationSeries(NamedTuple):
        """The series of values for the simulation."""

        median_uplift: ListLike[T_Value]
        min_uplift: ListLike[T_Value]
        max_uplift: ListLike[T_Value]
        partitions: ListLike[T_Value]
        frequencies: ListLike[T_Value]

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
        categorical = simulation.partitioning.is_categorical
        simulation_series = self._get_simulation_series()
        self._style.draw_uplift(
            feature_name=simulation.feature_name,
            target_name=simulation.target_name,
            min_percentile=simulation.min_percentile,
            max_percentile=simulation.max_percentile,
            categorical=categorical,
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            median_uplift=simulation_series.median_uplift,
            min_uplift=simulation_series.min_uplift,
            max_uplift=simulation_series.max_uplift,
        )
        return None

    def _draw_histogram(self) -> None:
        # draw the histogram of the simulation values
        categorical = self._model.partitioning.is_categorical
        simulation_series = self._get_simulation_series()
        self._style.draw_histogram(
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            categorical=categorical,
        )
        return None

    def _get_simulation_series(self) -> SimulationSeries:
        # return the simulation series for median uplift, min uplift, max uplift,
        # partitions and frequencies
        # If categorical is true, the series are sorted
        # increasingly with respect to order of the series median uplift.
        # If categorical is false the original order of the series is not changed.
        simulation: UnivariateSimulation = self._model

        simulation_lists = [
            simulation.median_uplift,
            simulation.min_uplift,
            simulation.max_uplift,
            simulation.partitioning.partitions(),
            simulation.partitioning.frequencies(),
        ]

        if simulation.partitioning.is_categorical:
            print("been here")
            simulation_lists = zip(*sorted(zip(*simulation_lists), key=lambda x: x[0]))
        return self.SimulationSeries(*simulation_lists)
