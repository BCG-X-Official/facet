"""
Univariate simulation of target uplift.
"""

from typing import *

import numpy as np
import pandas as pd

from gamma import ListLike
from gamma.model.prediction import PredictorFitCV
from gamma.sklearndf.transformation import FunctionTransformerDF
from gamma.yieldengine.partition import Partitioning, T_Number


class UnivariateSimulation:
    """
    Summary result of a univariate simulation.
    """

    def __init__(
        self,
        feature_name: str,
        target_name: str,
        partitioning: Partitioning,
        median_uplift: ListLike[T_Number],
        min_uplift: ListLike[T_Number],
        max_uplift: ListLike[T_Number],
        min_percentile: float,
        max_percentile: float,
    ):
        self._feature_name = feature_name
        self._target_name = target_name
        self._partitioning = partitioning
        self._median_uplift = median_uplift
        self._min_uplift = min_uplift
        self._max_uplift = max_uplift
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    @property
    def feature_name(self) -> str:
        return self._feature_name

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def partitioning(self) -> Partitioning:
        return self._partitioning

    @property
    def median_uplift(self) -> ListLike[T_Number]:
        return self._median_uplift

    @property
    def min_uplift(self) -> ListLike[T_Number]:
        return self._min_uplift

    @property
    def max_uplift(self) -> ListLike[T_Number]:
        return self._max_uplift

    @property
    def min_percentile(self) -> float:
        return self._min_percentile

    @property
    def max_percentile(self) -> float:
        return self._max_percentile


class UnivariateSimulator:
    """
    Simulate target uplift of one feature change.

    Given a fitted model and a feature of the model,
    :meth:`UnivariateSimulator.simulate_feature` computes the prediction uplift
    if one would set the feature to a constant value for each sample in each test split.

    Aggregated percentiles uplift are then computed by
    :meth:`UnivariateSimulator.aggregate_simulation_results`.

    :param model_fit: fitted model used for the simulation
    :param min_percentile: lower bound of the confidence interval (default: 2.5)
    :param max_percentile: upper bound of the confidence interval (default: 97.5)
"""

    F_SPLIT_ID = "split_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_TARGET_CHANGE = "relative_target_change"

    def __init__(
        self,
        model_fit: PredictorFitCV,
        min_percentile: float = 2.5,
        max_percentile: float = 97.5,
    ):
        self._model_fit = model_fit
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
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    @property
    def model_fit(self) -> PredictorFitCV:
        """The fitted model used for the simulation."""
        return self._model_fit

    @property
    def min_percentile(self) -> float:
        return self._min_percentile

    @property
    def max_percentile(self) -> float:
        return self._max_percentile

    def simulate_feature(
        self, feature_name: str, partitioning: Partitioning
    ) -> UnivariateSimulation:
        """
        Simulate average yield uplift for the given feature and for each partition in
        the given partitioning.

        :param feature_name: the feature of the simulation
        :param partitioning: the partitioning used for the simulation
        """
        simulated_values = partitioning.partitions()
        prediction_uplift = self._aggregate_simulation_results(
            results_per_split=(
                self._simulate_feature_with_values(
                    feature_name=feature_name, simulated_values=simulated_values
                )
            )
        )
        return UnivariateSimulation(
            feature_name=feature_name,
            target_name=self._model_fit.sample.target_name,
            partitioning=partitioning,
            median_uplift=prediction_uplift.iloc[:, 1].values,
            min_uplift=prediction_uplift.iloc[:, 0].values,
            max_uplift=prediction_uplift.iloc[:, 2].values,
            min_percentile=self._min_percentile,
            max_percentile=self._max_percentile,
        )

    def _simulate_feature_with_values(
        self, feature_name: str, simulated_values: Iterable[Any]
    ) -> pd.DataFrame:
        """
        Run a simulation on a feature.

        For each combination of split_id and feature value the uplift (in % as a
        number between 0 and 1) of the target is computed. It is the uplift between
        predictions on the sample where the `feature_name` column is set to the
        given value, compared to the predictions on the original sample.

        :param feature_name: name of the feature to use in the simulation
        :param simulated_values: values to use in the simulation
        :return: dataframe with three columns: `split_id`, `parameter_value` and
          `relative_target_change`.
        """
        if feature_name not in self.model_fit.sample.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        def _simulate_values() -> Generator[Tuple[int, Any, float], None, None]:
            for value in simulated_values:
                # replace the simulated column with a constant value
                synthetic_sample = FunctionTransformerDF(
                    func=lambda x: (x.assign(**{feature_name: value})), validate=False
                ).fit_transform_sample(self.model_fit.sample)

                fit_for_syn_sample = self.model_fit.copy_with_sample(
                    sample=synthetic_sample
                )

                for split_id in range(self.model_fit.n_splits):
                    predictions_for_split_hist: pd.Series = (
                        self.model_fit.predictions_for_split(split_id=split_id)
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
                UnivariateSimulator.F_SPLIT_ID,
                UnivariateSimulator.F_PARAMETER_VALUE,
                UnivariateSimulator.F_RELATIVE_TARGET_CHANGE,
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
            results_per_split.drop(columns=UnivariateSimulator.F_SPLIT_ID)
            .groupby(by=UnivariateSimulator.F_PARAMETER_VALUE)[
                UnivariateSimulator.F_RELATIVE_TARGET_CHANGE
            ]
            .agg(
                [
                    percentile(p)
                    for p in (self._min_percentile, 50, self._max_percentile)
                ]
            )
        )
