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
Univariate simulation of target uplift.
"""
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd

from gamma import ListLike
from gamma.model.fitcv import ClassifierFitCV, PredictorFitCV, RegressorFitCV
from gamma.sklearndf.transformation import FunctionTransformerDF
from gamma.yieldengine.partition import Partitioning, T_Number

T_PredictiveFitCV = TypeVar("T_PredictiveFitCV", bound=PredictorFitCV)


class UnivariateSimulation:
    """
    Summary result of a univariate simulation.

    :param feature_name: name of the feature on which the simulation is made
    :param target_name: name of the target
    :param partitioning: the partition of ``feature_name`` used for the simulation
    :param median_change: the median change values
    :param min_change:  the low percentile change values
    :param max_change: the high percentile change values
    :param min_percentile: the percentile used to compute ``min_uplift``. Must be a
      number between 0 and 100
    :param max_percentile: the percentile used to compute ``max_uplift``. Must be a
      number between 0 and 100
    """

    def __init__(
        self,
        feature_name: str,
        target_name: str,
        partitioning: Partitioning,
        median_change: ListLike[T_Number],
        min_change: ListLike[T_Number],
        max_change: ListLike[T_Number],
        min_percentile: float,
        max_percentile: float,
    ):
        self._feature_name = feature_name
        self._target_name = target_name
        self._partitioning = partitioning
        self._median_change = median_change
        self._min_change = min_change
        self._max_change = max_change
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    @property
    def feature_name(self) -> str:
        """Name of the feature on which the simulation is made."""
        return self._feature_name

    @property
    def target_name(self) -> str:
        """Name of the target."""
        return self._target_name

    @property
    def partitioning(self) -> Partitioning:
        """The partition of ``feature_name`` used for the simulation."""
        return self._partitioning

    @property
    def median_change(self) -> ListLike[T_Number]:
        """Median average change determined by a simulation."""
        return self._median_change

    @property
    def min_change(self) -> ListLike[T_Number]:
        """
        Minimum average change, at the lower end of the confidence interval,
        determined by a simulation.
        """
        return self._min_change

    @property
    def max_change(self) -> ListLike[T_Number]:
        """
        Minimum average change, at the lower end of the confidence interval,
        determined by a simulation.
        """
        return self._max_change

    @property
    def min_percentile(self) -> float:
        """
        Percentile of the lower end of thw confidence interval.
        """
        return self._min_percentile

    @property
    def max_percentile(self) -> float:
        """
        Percentile of the upper end of thw confidence interval.
        """
        return self._max_percentile


class UnivariateSimulator(Generic[T_PredictiveFitCV], ABC):
    """
    Predicts a change in outcome for a range of values for a given feature,
    using predictions of a fitted model. Works with collections of models fitted to
    different data splits, therefore can determine confidence intervals for the
    predicted values.

    Typical simulated outcomes are target variables of regressions, or probabilities
    of classifications.

    :param models: fitted models (based on a cross-validation strategy) used to
        predict changes in outcome during the simulation
    :param min_percentile: lower bound of the confidence interval (default: 2.5)
    :param max_percentile: upper bound of the confidence interval (default: 97.5)
    """

    def __init__(
        self,
        models: T_PredictiveFitCV,
        min_percentile: float = 2.5,
        max_percentile: float = 97.5,
    ):

        if not 0 <= min_percentile <= 100:
            raise ValueError(
                f"arg min_percentile={min_percentile} must be in the range"
                "from 0 to 100"
            )
        if not 0 <= max_percentile <= 100:
            raise ValueError(
                f"arg max_percentile={max_percentile} must be in the range"
                "from 0 to 1"
            )

        self._max_percentile = max_percentile
        self._min_percentile = min_percentile
        self._models = models

    @property
    def models(self) -> T_PredictiveFitCV:
        """The fitted models used for the simulation."""
        return self._models

    @property
    def min_percentile(self) -> float:
        """
        Percentile of the lower end of the confidence interval.
        """
        return self._min_percentile

    @property
    def max_percentile(self) -> float:
        """
        Percentile of the upper end of the confidence interval.
        """
        return self._max_percentile

    @abstractmethod
    def simulate_feature(self, feature_name: str, partitioning: Partitioning):
        """
        Simulate average target uplift for the given feature and for each partition in
        the given partitioning.

        :param feature_name: the feature of the simulation
        :param partitioning: the partitioning used for the simulation
        """
        pass


class UnivariateProbabilitySimulator(UnivariateSimulator[ClassifierFitCV]):
    def simulate_feature(self, feature_name: str, partitioning: Partitioning):
        raise NotImplementedError(
            "simulation of average change in probability will be included in a future "
            "release"
        )


class UnivariateUpliftSimulator(UnivariateSimulator[RegressorFitCV]):
    F_SPLIT_ID = "split_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_TARGET_CHANGE = "relative_target_change"

    def simulate_feature(
        self, feature_name: str, partitioning: Partitioning
    ) -> UnivariateSimulation:
        """
        Simulate average target uplift for the given feature and for each partition in
        the given partitioning.

        :param feature_name: the feature of the simulation
        :param partitioning: the partitioning used for the simulation
        """
        simulated_values = partitioning.partitions()
        predicted_change = self._aggregate_simulation_results(
            results_per_split=(
                self._simulate_feature_with_values(
                    feature_name=feature_name, simulated_values=simulated_values
                )
            )
        )
        return UnivariateSimulation(
            feature_name=feature_name,
            target_name=self.models.sample.target_name,
            partitioning=partitioning,
            median_change=predicted_change.iloc[:, 1].values,
            min_change=predicted_change.iloc[:, 0].values,
            max_change=predicted_change.iloc[:, 2].values,
            min_percentile=self._min_percentile,
            max_percentile=self._max_percentile,
        )

    def _simulate_feature_with_values(
        self, feature_name: str, simulated_values: Iterable[Any]
    ) -> pd.DataFrame:
        """
        Run a simulation on a feature.

        For each combination of split_id and feature value the uplift (in % as a
        number between -1 and 1) of the target is computed. It is the uplift between
        predictions on the sample where the `feature_name` column is set to the
        given value, compared to the predictions on the original sample.

        :param feature_name: name of the feature to use in the simulation
        :param simulated_values: values to use in the simulation
        :return: dataframe with three columns: `split_id`, `parameter_value` and
          `relative_target_change`.
        """
        if feature_name not in self.models.sample.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        def _simulate_values() -> Generator[Tuple[int, Any, float], None, None]:
            sample = self.models.sample
            feature_dtype = sample.features.loc[:, feature_name].dtype
            for value in simulated_values:
                # replace the simulated column with a constant value
                synthetic_sample = sample.replace_features(
                    FunctionTransformerDF(
                        func=lambda x: (
                            x.assign(**{feature_name: value}).astype(
                                {feature_name: feature_dtype}
                            )
                        ),
                        validate=False,
                    ).fit_transform(X=sample.features, y=sample.target)
                )

                fit_for_syn_sample = self.models.copy_with_sample(
                    sample=synthetic_sample
                )

                for split_id in range(self.models.n_splits):
                    predictions_for_split_hist: pd.Series = (
                        self.models.predictions_for_split(split_id=split_id)
                    )

                    predictions_for_split_syn: pd.Series = (
                        fit_for_syn_sample.predictions_for_split(split_id=split_id)
                    )

                    relative_target_change = (
                        predictions_for_split_syn.mean(axis=0)
                        / predictions_for_split_hist.mean(axis=0)
                    ) - 1

                    yield split_id, value, relative_target_change

        return pd.DataFrame(
            data=_simulate_values(),
            columns=[
                UnivariateUpliftSimulator.F_SPLIT_ID,
                UnivariateUpliftSimulator.F_PARAMETER_VALUE,
                UnivariateUpliftSimulator.F_RELATIVE_TARGET_CHANGE,
            ],
        )

    def _aggregate_simulation_results(
        self, results_per_split: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate uplift values computed by `simulate_feature`.

        For each parameter value, the percentile of uplift values (in the
        `relative_yield_change` column) are computed.

        :param results_per_split: dataframe with columns `split_id`, `parameter_value`\
          and `relative_yield_change`
        :return: dataframe with 3 columns `percentile_<min>`, `percentile_50`,
          `percentile<max>` where min/max are the min and max percentiles
        """

        def percentile(n: int) -> Callable[[float], float]:
            """
            Return the function computed the n-th percentile.

            :param n: the percentile to compute; int between 0 and 100
            :return: the n-th percentile function
            """

            def percentile_(x: float):
                """n-th percentile function"""
                return np.percentile(x, n)

            percentile_.__name__ = "percentile_%s" % n
            return percentile_

        return (
            results_per_split.drop(columns=UnivariateUpliftSimulator.F_SPLIT_ID)
            .groupby(by=UnivariateUpliftSimulator.F_PARAMETER_VALUE)[
                UnivariateUpliftSimulator.F_RELATIVE_TARGET_CHANGE
            ]
            .agg(
                [
                    percentile(p)
                    for p in (self._min_percentile, 50, self._max_percentile)
                ]
            )
        )
