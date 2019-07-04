from typing import *

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

from yieldengine import Sample
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
        self, parameterized_feature: str, parameter_values: np.ndarray
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
        parameterized_feature: str, parameter_value
    ) -> FunctionTransformerDF:
        """

        :param parameterized_feature:
        :param parameter_value:
        :return:
        """
        def transform(x: pd.DataFrame) -> pd.DataFrame:
            x[parameterized_feature] = parameter_value
            return x

        return FunctionTransformerDF(func=transform, validate=False)

    @staticmethod
    def observed_feature_values(
        sample: Sample,
        feature_name: str,
        min_relative_frequency: float = 0.05,
        limit_observations: int = 20,
    ) -> np.ndarray:
        """
        Get an array of observed values for a particular feature

        :param feature_name: name of the feature
        :param min_relative_frequency: the relative frequency with which a particular \
        feature value has to occur within the sample, for it to be selected. Not used \
        for non-discrete features or features with high variability (when no single \
        feature value occurs more than "min_relative_frequency" times)
        :param limit_observations: how many observation-values to return at max.
        :return: a 1D numpy array with the selected feature values
        """

        # get the series of the feature and drop NAs
        feature_sr = sample.features.loc[:, feature_name].dropna()

        # get value counts
        times_observed_sr = feature_sr.value_counts()

        # get relative frequency for each feature value and filter using
        # min_relative_frequency, then determines the limit_observations most frequent
        # observations
        observed_filtered = (
            times_observed_sr.loc[
                times_observed_sr / times_observed_sr.sum() >= min_relative_frequency
            ]
            .sort_values(ascending=False)
            .index[:limit_observations]
        )

        # if the feature is not numeric, always only use frequency based approach
        if not is_numeric_dtype(feature_sr):
            return observed_filtered

        # feature is numeric and either
        #  a) feature is non-discrete/non-int datatype
        #  b) above approach did not return any feature values (i.e. because of too
        #  much variation even in an all integer feature)
        # --> go with approach below
        elif len(observed_filtered) == 0 or (
            # the series includes float values that do not represent whole numbers
            not np.all(feature_sr == feature_sr.astype(int))
        ):
            # get a sorted array of all unique values for the feature
            unique_values_sorted: np.ndarray = feature_sr.copy().unique()
            unique_values_sorted.sort()
            # are there more unique-values than allowed by the passed limit?
            if len(unique_values_sorted) > limit_observations:
                # use np.linspace to spread out array indices evenly within bounds
                value_samples = np.linspace(
                    0, len(unique_values_sorted) - 1, limit_observations
                ).astype(int)
                # return "sampled" feature values out of all unique feature values
                return unique_values_sorted[value_samples]
            else:
                # return all unique values, since they are within limit bound
                return unique_values_sorted
        else:
            return observed_filtered
