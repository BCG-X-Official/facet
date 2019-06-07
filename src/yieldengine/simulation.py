from typing import *

import numpy as np
import pandas as pd

from yieldengine import Sample
from yieldengine.model.prediction import PredictorCV


class UnivariateSimulation:
    __slots__ = ["_predictor"]

    F_FOLD_ID = "fold_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_YIELD_CHANGE = "relative_yield_change"

    def __init__(self, predictor: PredictorCV):
        self._predictor = predictor

    @property
    def predictor(self) -> PredictorCV:
        return self._predictor

    def simulate_yield_change(
        self, parameterized_feature: str, parameter_values: np.ndarray
    ) -> pd.DataFrame:
        if parameterized_feature not in self.predictor.sample.feature_names:
            raise ValueError(f"Feature '{parameterized_feature}' not in sample")

        results = []

        self.predictor.fit()

        for parameter_value in parameter_values:
            sample_df = self.predictor.sample.observations.copy()
            sample_df.loc[:, parameterized_feature] = parameter_value

            synthetic_sample = Sample(
                observations=sample_df, target_name=self.predictor.sample.target_name
            )

            predictor_for_syn_sample = self.predictor.copy_with_sample(
                sample=synthetic_sample
            )

            predictor_for_syn_sample.fit()

            for fold_id in self.predictor.fold_ids:
                predictions_for_fold_hist: np.ndarray = self.predictor.predictions_for_fold(
                    fold_id=fold_id
                )

                predictions_for_fold_syn: np.ndarray = predictor_for_syn_sample.predictions_for_fold(
                    fold_id=fold_id
                )

                relative_yield_change = (
                    predictions_for_fold_syn.mean(axis=0)
                    / predictions_for_fold_hist.mean(axis=0)
                ) - 1

                results.append(
                    {
                        UnivariateSimulation.F_FOLD_ID: fold_id,
                        UnivariateSimulation.F_PARAMETER_VALUE: parameter_value,
                        UnivariateSimulation.F_RELATIVE_YIELD_CHANGE: relative_yield_change,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def aggregate_simulated_yield_change(
        foldwise_results: pd.DataFrame, percentiles: List[int]
    ):
        def percentile(n: int):
            def percentile_(x: float):
                return np.percentile(x, n)

            percentile_.__name__ = "percentile_%s" % n
            return percentile_

        return (
            foldwise_results.drop(columns=UnivariateSimulation.F_FOLD_ID)
            .groupby(by=UnivariateSimulation.F_PARAMETER_VALUE)
            .agg([percentile(p) for p in percentiles])
        )
