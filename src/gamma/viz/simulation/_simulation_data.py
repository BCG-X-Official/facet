"""
Simulation data
"""
from typing import List, Optional, Tuple, TypeVar

import pandas as pd

from gamma.model.prediction import PredictorFitCV
from gamma.yieldengine.partition import (
    ContinuousRangePartitioning,
    NumericType,
    RangePartitioning,
)
from gamma.yieldengine.simulation import UnivariateSimulation

T_RangePartitioning = TypeVar("T_RangePartitioning", bound=RangePartitioning)

DEFAULT_PARTITIONING = ContinuousRangePartitioning


class SimulationData:
    """
    Simulation on the target predictions of setting the values of a feature.

    Given the fitted model :attr:`~SimulatedData.model`, and the feature to simulate on
    :attr:`~SimulatedData.feature`, a partition of the feature values is computed
    which is then used to run a univariate simulation.

    The parameters of the partitioning can be updated with
    :meth:`~SimulationData.update`.

    :param feature: the feature used to make the simulation
    :param max_partitions: the maximum number of values from ``feature`` used to make
      the simulation. The default value ``None`` will then use a maximum of 20
      partitions
    :param lower_bound: lower bound on the values of ``feature`` used in the
    simulation. The default value ``None`` will not impose any lower bound.
    :param upper_bound: upper bound on the values of ``feature`` used in the
      simulation. The default value ``None`` does not impose any upper bound.
    :param partition_type: the type of partition used. When ``None`` (default)
    :class:`~gamma.yieldengine.partition.ContinuousRangePartitioning` is used.
    :param percentiles: the percentiles used for low, middle and high simulated
      predictions
    """

    def __init__(
        self,
        model: PredictorFitCV,
        feature: str,
        max_partitions: Optional[int] = None,
        lower_bound: Optional[NumericType] = None,
        upper_bound: Optional[NumericType] = None,
        partition_type: Optional[T_RangePartitioning] = None,
        percentiles: Optional[Tuple[int, int, int]] = (10, 50, 90),
    ) -> None:
        self._model = model
        self._sample = self._model.sample
        self._feature = feature
        self._max_partitions = max_partitions
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        if partition_type:
            self._partition_type = partition_type
        else:
            self._partition_type = DEFAULT_PARTITIONING
        self._percentiles = percentiles
        self._simulation = UnivariateSimulation(model_fit=self._model)
        self._target_name = self._sample.target_name
        self._partitioning = None
        self._results_per_split = None
        self._prediction_uplifts = None
        self._high_confidence_uplift = None
        self._middle_confidence_uplift = None
        self._low_confidence_uplift = None
        self._run()

    def _run(self) -> None:
        # run the simulation and compute the confidence values for the uplift

        # partition the feature values
        self._partitioning = self._get_partitioning()

        # compute the results for each split and each feature value in the partitions
        self._results_per_split = self._simulation.simulate_feature(
            feature_name=self.feature, feature_values=self.partitioning.partitions()
        )

        # compute the aggregated prediction percentile: low, middle and high
        # percentiles
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

    def _get_partitioning(self) -> RangePartitioning:
        # Return the class used to partition the simulated feature
        return self._partition_type(
            values=self._sample.features[self._feature],
            lower_bound=self._lower_bound,
            upper_bound=self._upper_bound,
            max_partitions=self._max_partitions,
        )

    def update(
        self,
        feature: Optional[str] = None,
        max_partitions: Optional[int] = None,
        lower_bound: Optional[NumericType] = None,
        upper_bound: Optional[NumericType] = None,
        partition_type: Optional[T_RangePartitioning] = None,
        percentiles: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Update the partition parameters and run the simulation.

        :param feature: the feature used to make the simulation
        :param max_partitions: the maximum number of values from ``feature``
          used to make
          the simulation. The default value ``None`` will then use a maximum of 20
          partitions
        :param lower_bound: lower bound on the values of ``feature`` used in the
        simulation. The default value ``None`` will not impose any lower bound.
        :param upper_bound: upper bound on the values of ``feature`` used in the
          simulation. The default value ``None`` does not impose any upper bound.
        :param partition_type: the type of partition used. When ``None`` (default)
          :class:`~gamma.yieldengine.partition.ContinuousRangePartitioning` is used.
        :param percentiles: the percentiles used for low, middle and high simulated
          predictions
        """
        if feature:
            self._feature = feature
        if max_partitions:
            self._max_partitions = max_partitions
        if lower_bound:
            self._lower_bound = lower_bound
        if upper_bound:
            self._upper_bound = upper_bound
        if partition_type:
            self._partition_type = partition_type
        if percentiles:
            self._percentiles = percentiles
        self._run()
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
    def target_name(self) -> str:
        """The target name."""
        return self._target_name

    @property
    def results_per_split(self) -> pd.DataFrame:
        """
        Results of the simulation per split.

        This is the output of
        :meth:`gamma.yieldengine.simulation.UnivariateSimulation.simulate_feature`.

        :return: for each combination of split, row and simulated value, it gives the
           prediction
        """
        return self._results_per_split

    @property
    def partitioning(self) -> ContinuousRangePartitioning:
        """The partitioning used for the simulated feature values."""
        return self._partitioning

    @property
    def low_confidence_uplift(self) -> Tuple[List[NumericType], List[NumericType]]:
        """
        Low confidence values for the uplift simulation.

        :return: (x, y) where x is the list of simulated feature values and y is the
        list of low percentile uplift
        """
        return self._low_confidence_uplift

    @property
    def middle_confidence_uplift(self) -> Tuple[List[NumericType], List[NumericType]]:
        """
        Middle confidence for the uplift simulation.

        :return: (x, y) where x is the list of simulated feature values and y is the
        list of middle percentile uplift
        """
        return self._middle_confidence_uplift

    @property
    def high_confidence_uplift(self) -> Tuple[List[NumericType], List[NumericType]]:
        """
        High confidence for the uplift simulation.

        :return: (x, y) where x is the list of simulated feature values and y is the
        list of middle percentile uplift
        """
        return self._high_confidence_uplift
