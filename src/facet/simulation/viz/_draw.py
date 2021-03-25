"""
Visualizations of simulation results.
"""

from typing import Any, Iterable, Optional, Sequence, Tuple, Type, TypeVar, Union

from pytools.api import AllTracker, inheritdoc
from pytools.viz import Drawer

from ._style import SimulationMatplotStyle, SimulationReportStyle, SimulationStyle
from facet.simulation import UnivariateSimulationResult

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


@inheritdoc(match="[see superclass]")
class SimulationDrawer(Drawer[UnivariateSimulationResult, SimulationStyle]):
    """
    Draws the result of a univariate simulation, represented by a
    :class:`.UnivariateSimulationResult` object.
    """

    #: if ``True``, plot the histogram of observed values for the feature being
    #: simulated; if ``False``, do not plot the histogram
    histogram: bool

    def __init__(
        self,
        style: Optional[Union[SimulationStyle, str]] = None,
        histogram: bool = True,
    ) -> None:
        """
        :param histogram: if ``True``, plot the histogram of observed values for the
            feature being simulated; if ``False``, do not plot the histogram (default:
            ``True``).
        """
        super().__init__(style=style)
        self.histogram = histogram

    __init__.__doc__ = Drawer.__init__.__doc__ + __init__.__doc__

    def draw(
        self, data: UnivariateSimulationResult, *, title: Optional[str] = None
    ) -> None:
        """
        Draw the simulation chart.

        :param data: the univariate simulation to draw
        :param title: the title of the chart (optional, defaults to the name of the
            simulated feature)
        """
        if title is None:
            title = f"Simulation: {data.feature_name}"
        super().draw(data=data, title=title)

    @classmethod
    def get_style_classes(cls) -> Iterable[Type[SimulationStyle]]:
        """[see superclass]"""

        return [
            SimulationMatplotStyle,
            SimulationReportStyle,
        ]

    def _draw(self, data: UnivariateSimulationResult) -> None:
        # If the partitioning of the simulation is categorical, sort partitions in
        # ascending order of the median output
        simulation_result: Tuple[
            Sequence[float],
            Sequence[float],
            Sequence[float],
            Sequence[Any],
            Sequence[int],
        ] = (
            data.outputs_median().to_list(),
            data.outputs_lower_bound().to_list(),
            data.outputs_upper_bound().to_list(),
            data.partitioner.partitions_,
            data.partitioner.frequencies_,
        )

        if data.partitioner.is_categorical:
            # for categorical features, sort the categories by the median uplift
            simulation_result = tuple(
                *zip(*sorted(zip(*simulation_result), key=lambda x: x[0]))
            )

        # draw the graph with the uplift curves
        self.style.draw_uplift(
            feature_name=data.feature_name,
            output_name=data.output_name,
            output_unit=data.output_unit,
            outputs_median=simulation_result[0],
            outputs_lower_bound=simulation_result[1],
            outputs_upper_bound=simulation_result[2],
            baseline=data.baseline,
            confidence_level=data.confidence_level,
            partitions=simulation_result[3],
            frequencies=simulation_result[4],
            is_categorical_feature=data.partitioner.is_categorical,
        )

        if self.histogram:
            # draw the histogram of the simulation values
            self.style.draw_histogram(
                partitions=simulation_result[3],
                frequencies=simulation_result[4],
                is_categorical_feature=data.partitioner.is_categorical,
            )


__tracker.validate()
