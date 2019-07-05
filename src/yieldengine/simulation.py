from typing import *

import numpy as np
import pandas as pd

from yieldengine.model.prediction import PredictorCV
from yieldengine.preprocessing import FunctionTransformerDF


class UnivariateSimulation:
    __slots__ = ["_predictor"]

    F_SPLIT_ID = "split_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_YIELD_CHANGE = "relative_yield_change"

    def __init__(self, predictor: PredictorCV):
        self._predictor = predictor

    @property
    def predictor(self) -> PredictorCV:
        return self._predictor

    def simulate_yield_change(
        self, parameterized_feature: str, parameter_values: Iterable[Any]
    ) -> pd.DataFrame:
        if parameterized_feature not in self.predictor.sample.feature_names:
            raise ValueError(f"Feature '{parameterized_feature}' not in sample")

        results = []

        self.predictor.fit()

        for parameter_value in parameter_values:
            simul_transformer = UnivariateSimulation.make_column_replacing_transformer(
                parameterized_feature=parameterized_feature,
                parameter_value=parameter_value,
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

                relative_yield_change = (
                    predictions_for_split_syn.mean(axis=0)
                    / predictions_for_split_hist.mean(axis=0)
                ) - 1

                relative_yield_change_ = (
                    split_id,
                    parameter_value,
                    relative_yield_change,
                )
                results.append(relative_yield_change_)

        return pd.DataFrame(
            results,
            columns=[
                UnivariateSimulation.F_SPLIT_ID,
                UnivariateSimulation.F_PARAMETER_VALUE,
                UnivariateSimulation.F_RELATIVE_YIELD_CHANGE,
            ],
        )

    @staticmethod
    def aggregate_simulated_yield_change(
        results_per_split: pd.DataFrame, percentiles: List[int]
    ) -> pd.DataFrame:
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
        parameterized_feature: str, parameter_value
    ) -> FunctionTransformerDF:
        def transform(x: pd.DataFrame) -> pd.DataFrame:
            x[parameterized_feature] = parameter_value
            return x

        return FunctionTransformerDF(func=transform, validate=False)
