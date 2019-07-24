"""
Univariate simulation of target uplift.
"""

from typing import *

import numpy as np
import pandas as pd

from gamma.model.prediction import PredictorFitCV
from gamma.sklearndf.transformation import FunctionTransformerDF


class UnivariateSimulator:
    """
    Simulate target uplift of one feature change.

    Given a fitted model and a feature of the model,
    :meth:`UnivariateSimulator.simulate_feature` computes the prediction uplift
    if one would set the feature to a constant value for each sample in each test split.

    Aggregated percentiles uplift are then computed by
    :meth:`UnivariateSimulator.aggregate_simulation_results`.

    :param model_fit: fitted model used for the simulation
    """

    __slots__ = ["_model_fit"]

    F_SPLIT_ID = "split_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_TARGET_CHANGE = "relative_target_change"

    def __init__(self, model_fit: PredictorFitCV):
        self._model_fit = model_fit

    @property
    def model_fit(self) -> PredictorFitCV:
        """The fitted model used for the simulation."""
        return self._model_fit

    def simulate_feature(
        self, feature_name: str, feature_values: Iterable[Any]
    ) -> pd.DataFrame:
        """
        Run a simulation on a feature.

        For each combination of split_id and feature value the uplift (in % as a
        number between 0 and 1) of the target is computed. It is the uplift between
        predictions on the sample where the `feature_name` column is set to the
        given value, compared to the predictions on the original sample.

        :param feature_name: name of the feature to use in the simulation
        :param feature_values: values to use in the simulation
        :return: dataframe with three columns: `split_id`, `parameter_value` and
          `relative_target_change`.
        """
        if feature_name not in self.model_fit.sample.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        results = []

        for value in feature_values:
            # Replace the simulated column with a constant value.
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

                results.append((split_id, value, relative_target_change))

        return pd.DataFrame(
            results,
            columns=[
                UnivariateSimulator.F_SPLIT_ID,
                UnivariateSimulator.F_PARAMETER_VALUE,
                UnivariateSimulator.F_RELATIVE_TARGET_CHANGE,
            ],
        )

    @staticmethod
    def aggregate_simulation_results(
        results_per_split: pd.DataFrame, percentiles: Iterable[int]
    ) -> pd.DataFrame:
        """
        Aggregate uplift values computed by `simulate_feature`.

        For each parameter value, the percentile of uplift values (in the
        `relative_yield_change` column) are computed.

        :param results_per_split: dataframe with columns `split_id`, `parameter_value`\
          and `relative_yield_change`
        :param percentiles: the list of percentiles
        :return: dataframe with columns percentile_<p> where p goes through the list
          `percentiles` and whose index is given by the parameter values
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
            .groupby(by=UnivariateSimulator.F_PARAMETER_VALUE)
            .agg([percentile(p) for p in percentiles])
        )
