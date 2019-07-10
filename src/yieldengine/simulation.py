from typing import *

import numpy as np
import pandas as pd

from yieldengine.model.prediction import PredictorCV
from yieldengine.preprocessing import FunctionTransformerDF


class UnivariateSimulation:
    __slots__ = ["_predictor"]

    F_SPLIT_ID = "split_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_TARGET_CHANGE = "relative_target_change"

    def __init__(self, predictor: PredictorCV):
        self._predictor = predictor

    @property
    def predictor(self) -> PredictorCV:
        return self._predictor

    def simulate_feature(
        self, feature_name: str, feature_values: Iterable[Any]
    ) -> pd.DataFrame:
        """
        Run a simulation on a feature.

        For each combination of split_id and parameter_value the uplift (in % as a
        number between 0 and 1) of the target is computed. It is the uplift between
        predictions on the sample where the `parametrized_feature` is set to the
        given value, compared to the predictions on the original sample.

        :param parameterized_feature: name of the feature to use in the simulation
        :param parameter_values: values to use in the simulation
        :return: dataframe with three columns: `split_id`, `parameter_value` and
        `relative_yield_change`.
        """
        if feature_name not in self.predictor.sample.feature_names:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        results = []

        self.predictor.fit()

        for value in feature_values:
            simul_transformer = UnivariateSimulation.make_column_replacing_transformer(
                feature_name=feature_name, feature_value=value
            )

            synthetic_sample = simul_transformer.fit_transform_sample(
                self.predictor.sample
            )

            predictor_for_syn_sample = self.predictor.copy_with_sample(
                sample=synthetic_sample
            )

            predictor_for_syn_sample.fit()

            for split_id in self.predictor.split_ids:
                predictions_for_split_hist: pd.Series = (
                    self.predictor.predictions_for_split(split_id=split_id)
                )

                predictions_for_split_syn: pd.Series = (
                    predictor_for_syn_sample.predictions_for_split(split_id=split_id)
                )

                relative_target_change = (
                    predictions_for_split_syn.mean(axis=0)
                    / predictions_for_split_hist.mean(axis=0)
                ) - 1

                relative_target_change_ = (split_id, value, relative_target_change)
                results.append(relative_target_change_)

        return pd.DataFrame(
            results,
            columns=[
                UnivariateSimulation.F_SPLIT_ID,
                UnivariateSimulation.F_PARAMETER_VALUE,
                UnivariateSimulation.F_RELATIVE_TARGET_CHANGE,
            ],
        )

    @staticmethod
    def aggregate_simulation_results(
        results_per_split: pd.DataFrame, percentiles: List[int]
    ) -> pd.DataFrame:
        """
        Aggregate the upflift values computed by `simulate_yield_changes`.

        For each parameter value, the percentile of uplift values (in the
        `relative_yield_change` column) are computed.

        :param results_per_split: dataframe with columns `split_id`, `parameter_value`\
         and `relative_yield_change`.
        :param percentiles: the list of percentiles
        :return: dataframe with columns percentile_<p> where p goes through the list
        `percentiles` and whose index is given by the parameter values.
        """

        def percentile(n: int):
            def percentile_(x: float):
                return np.percentile(x, n)

            percentile_.__name__ = "percentile_%s" % n
            return percentile_

        return (
            results_per_split.drop(columns=UnivariateSimulation.F_SPLIT_ID)
            .groupby(by=UnivariateSimulation.F_PARAMETER_VALUE)
            .agg([percentile(p) for p in percentiles])
        )

    @staticmethod
    def make_column_replacing_transformer(
        feature_name: str, feature_value
    ) -> FunctionTransformerDF:
        def transform(x: pd.DataFrame) -> pd.DataFrame:
            x[feature_name] = feature_value
            return x

        return FunctionTransformerDF(func=transform, validate=False)
