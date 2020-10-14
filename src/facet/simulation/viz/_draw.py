"""
Visualizations of simulation results.
"""

from typing import (
    Generic,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from pytools.api import AllTracker
from pytools.viz import Drawer

from ._style import SimulationMatplotStyle, SimulationReportStyle, SimulationStyle
from facet.simulation import UnivariateSimulation

__all__ = ["SimulationDrawer"]

#
# Type variables
#


T_Number = TypeVar("T_Number", int, float)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class _SimulationSeries(NamedTuple, Generic[T_Number]):
    # A set of aligned series representing the simulation result
    values_median: Sequence[T_Number]
    values_min: Sequence[T_Number]
    values_max: Sequence[T_Number]
    partitions: Sequence[T_Number]
    frequencies: Sequence[T_Number]


class SimulationDrawer(Drawer[UnivariateSimulation, SimulationStyle]):
    """
    Draws the result of a univariate simulation, represented by a
    :class:`.UnivariateSimulation` object.
    """

    _STYLES = {"matplot": SimulationMatplotStyle, "text": SimulationReportStyle}

    def __init__(
        self, style: Union[SimulationStyle, str] = "matplot", histogram: bool = True
    ) -> None:
        """
        :param style: the style of the dendrogram; either as a \
            :class:`.SimulationStyle` instance, or as the name of a \
            default style. Permissible names are ``"matplot"`` for a style based on \
            `matplotlib`, and ``"text"`` for a text-based report to stdout \
            (default: ``"matplot"``)
        :param histogram: if ``True``, plot the histogram of observed values for the \
            feature being simulated; if ``False`` do not plot the histogram (default: \
            ``True``).
        """
        super().__init__(style=style)
        self._histogram = histogram

    def draw(self, data: UnivariateSimulation, title: Optional[str] = None) -> None:
        """
        Draw the simulation chart.
        :param data: the univariate simulation to draw
        :param title: the title of the chart (optional, defaults to the name of the \
            simulated feature)
        """
        if title is None:
            title = f"Simulation: {data.feature}"
        super().draw(data=data, title=title)

    @classmethod
    def _get_style_dict(cls) -> Mapping[str, Type[SimulationStyle]]:
        return SimulationDrawer._STYLES

    def _draw(self, data: UnivariateSimulation) -> None:
        # draw the simulation chart
        simulation_series = self._get_simulation_series(simulation=data)

        # draw the graph with the uplift curves
        self._style.draw_uplift(
            feature=data.feature,
            target=data.target,
            values_label=data.values_label,
            values_median=simulation_series.values_median,
            values_min=simulation_series.values_min,
            values_max=simulation_series.values_max,
            values_baseline=data.values_baseline,
            percentile_lower=data.percentile_lower,
            percentile_upper=data.percentile_upper,
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            is_categorical_feature=data.partitioner.is_categorical,
        )

        if self._histogram:
            # draw the histogram of the simulation values
            self._style.draw_histogram(
                partitions=simulation_series.partitions,
                frequencies=simulation_series.frequencies,
                is_categorical_feature=data.partitioner.is_categorical,
            )

    @staticmethod
    def _get_simulation_series(simulation: UnivariateSimulation) -> _SimulationSeries:
        # return the simulation series for median uplift, min uplift, max uplift,
        # partitions and frequencies
        # If the partitioning of the simulation is categorical, the series are
        # sorted in ascending order of the median uplift.
        # Otherwise, the simulation series are returned unchanged.

        simulation_series = _SimulationSeries(
            simulation.values_median,
            simulation.values_lower,
            simulation.values_upper,
            simulation.partitioner.partitions_,
            simulation.partitioner.frequencies_,
        )

        if simulation.partitioner.is_categorical:
            # for categorical features, sort the categories by the median uplift
            return _SimulationSeries(
                *zip(*sorted(zip(*simulation_series), key=lambda x: x[0]))
            )
        else:
            return simulation_series


__tracker.validate()
