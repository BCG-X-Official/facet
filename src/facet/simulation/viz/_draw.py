"""
Visualizations of simulation results.
"""

from typing import Iterable, Optional, Type, TypeVar, Union, cast

import pandas as pd

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

    __init__.__doc__ = cast(str, Drawer.__init__.__doc__) + cast(str, __init__.__doc__)

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

    def _draw(self, result: UnivariateSimulationResult) -> None:
        # If the partitioning of the simulation is categorical, sort partitions in
        # ascending order of the mean output
        simulation_result: pd.DataFrame = result.data.assign(
            frequencies=result.partitioner.frequencies_
        )

        if result.partitioner.is_categorical:
            # for categorical features, sort the categories by mean predictions
            simulation_result = simulation_result.sort_values(
                by=UnivariateSimulationResult.COL_MEAN
            )

        partitions = simulation_result.index.values
        frequencies = simulation_result.frequencies.values

        # draw the graph with the uplift curves
        self.style.draw_uplift(
            feature_name=result.feature_name,
            output_name=result.output_name,
            output_unit=result.output_unit,
            outputs_mean=(
                simulation_result.loc[:, UnivariateSimulationResult.COL_MEAN].values
            ),
            outputs_lower_bound=(
                simulation_result.loc[
                    :, UnivariateSimulationResult.COL_LOWER_BOUND
                ].values
            ),
            outputs_upper_bound=(
                simulation_result.loc[
                    :, UnivariateSimulationResult.COL_UPPER_BOUND
                ].values
            ),
            baseline=result.baseline,
            confidence_level=result.confidence_level,
            partitions=partitions,
            frequencies=frequencies,
            is_categorical_feature=result.partitioner.is_categorical,
        )

        if self.histogram:
            # draw the histogram of the simulation values
            self.style.draw_histogram(
                partitions=partitions,
                frequencies=frequencies,
                is_categorical_feature=result.partitioner.is_categorical,
            )


__tracker.validate()
