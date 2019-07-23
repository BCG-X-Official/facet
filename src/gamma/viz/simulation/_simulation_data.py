"""
Simulation data
"""
from typing import Iterable, Tuple

import pandas as pd

from gamma.model.prediction import PredictorFitCV
from gamma.yieldengine.partition import (
    ContinuousRangePartitioning,
    NumericType,
    Partitioning,
)
from gamma.yieldengine.simulation import UnivariateSimulation


class SimulationData:
    """
    Data from the simulation used for drawing.
    """

    def __init__(self, model: PredictorFitCV, feature: str) -> None:
        self._model = model
        self._sample = self._model.sample
        self._feature = feature
        self._simulation = UnivariateSimulation(model_fit=self._model)
        self._percentiles = [10, 50, 90]
        self._partitioning = None
        self._results_per_split = None
        self._prediction_uplifts = None
        self._high_confidence_uplift = None
        self._middle_confidence_uplift = None
        self._low_confidence_uplift = None
        self._run()

    def _run(self) -> None:
        self._partitioning = ContinuousRangePartitioning(
            self._sample.features[self._feature]
        )
        self._results_per_split = self._simulation.simulate_feature(
            feature_name=self.feature, feature_values=self.partitioning.partitions()
        )
        self._prediction_uplifts = self._simulation.aggregate_simulation_results(
            results_per_split=self._results_per_split, percentiles=self._percentiles
        )
        self._low_confidence_uplift = (
            self._prediction_uplifts.index.to_list(),
            self._prediction_uplifts.iloc[:, 0].to_list(),
        )
        self._middle_confidence_uplift = (
            self._prediction_uplifts.index.to_list(),
            self._prediction_uplifts.iloc[:, 1].to_list(),
        )
        self._high_confidence_uplift = (
            self._prediction_uplifts.index.to_list(),
            self._prediction_uplifts.iloc[:, 2].to_list(),
        )
        return None

    def get_feature(self) -> str:
        """The feature name used in the simulation."""
        return self._feature

    def set_feature(self, value: str) -> None:
        """Set the feature name and update the aggregated simulation predictions."""
        self._feature = value
        self._run()

    feature = property(fget=get_feature, fset=set_feature)

    @property
    def results_per_split(self) -> pd.DataFrame:
        """
        Results of the simulation per split.

        Output of
        :meth:`gamma.yieldengine.simulation.UnivariateSimulation.simulate_feature`.
        """
        return self._results_per_split

    @property
    def partitioning(self) -> Partitioning:
        """The partitioning used for the simulated feature values."""
        return self._partitioning

    @property
    def low_confidence_uplift(
        self
    ) -> Tuple[Iterable[NumericType], Iterable[NumericType]]:
        """Low confidence for the uplift simulation."""
        return self._low_confidence_uplift

    @property
    def middle_confidence_uplift(
        self
    ) -> Tuple[Iterable[NumericType], Iterable[NumericType]]:
        """Middle confidence for the uplift simulation."""
        return self._middle_confidence_uplift

    @property
    def high_confidence_uplift(
        self
    ) -> Tuple[Iterable[NumericType], Iterable[NumericType]]:
        """High confidence for the uplift simulation."""
        return self._high_confidence_uplift
