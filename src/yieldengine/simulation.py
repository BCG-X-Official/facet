from typing import *

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from yieldengine import Sample
from yieldengine.model.inspection import ModelInspector


class UnivariateSimulation:

    F_FOLD_ID = "fold_id"
    F_PARAMETER_VALUE = "parameter_value"
    F_RELATIVE_YIELD_CHANGE = "relative_yield_change"

    def __init__(
        self, cv: BaseCrossValidator, sample: Sample, inspector: ModelInspector
    ):
        self._sample = sample
        self._cv = cv
        self._inspector = inspector

    def simulate_yield_change(
        self, parameterized_feature: str, parameter_values: np.ndarray
    ) -> pd.DataFrame:
        if parameterized_feature not in self._sample.feature_names:
            raise ValueError(f"Feature '{parameterized_feature}' not in sample")

        results = []

        for fold_id, (train_indices, test_indices) in enumerate(
            self._cv.split(self._sample.features, self._sample.target)
        ):
            for parameter_value in parameter_values:
                pipeline = self._inspector.pipeline(fold_id)
                predictions_for_fold: np.ndarray = self._inspector.predictions_for_fold(
                    fold_id=fold_id
                )

                test_data_features = self._sample.select_observations(
                    numbers=test_indices
                ).features.copy()

                test_data_features.loc[:, parameterized_feature] = parameter_value

                predictions_simulated: np.ndarray = pipeline.predict(
                    X=test_data_features
                )

                relative_yield_change = (
                    predictions_simulated.mean(axis=0)
                    / predictions_for_fold.mean(axis=0)
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
