import numpy as np
import pandas as pd

from yieldengine import Sample

DEFAULT_MIN_RELATIVE_FREQUENCY = 0.05
DEFAULT_LIMIT_OBSERVATIONS = 20


def observed_categorical_feature_values(
    sample: Sample,
    feature_name: str,
    min_relative_frequency: float = DEFAULT_MIN_RELATIVE_FREQUENCY,
    limit_observations: int = DEFAULT_LIMIT_OBSERVATIONS,
) -> np.ndarray:
    """
    Get an array of observed values for a particular categorical feature

    :param sample: Sample the feature belongs to
    :param feature_name: name of the feature
    :param min_relative_frequency: the relative frequency with which a particular
    feature value has to occur within the sample, for it to be selected.
    :param limit_observations: how many observation-values to return at max.
    :return: a 1D numpy array with the selected feature values
    """
    # todo: decide if categoricals need special treatment vs. discrete
    return _sample_by_frequency(
        series=_select_feature(sample=sample, feature_name=feature_name),
        min_relative_frequency=min_relative_frequency,
        limit_observations=limit_observations,
    )


def observed_discrete_feature_values(
    sample: Sample,
    feature_name: str,
    min_relative_frequency: float = DEFAULT_MIN_RELATIVE_FREQUENCY,
    limit_observations: int = DEFAULT_LIMIT_OBSERVATIONS,
    fallback_to_range_based: bool = True,
) -> np.ndarray:
    """
    Get an array of observed values for a particular discrete feature

    :param sample: Sample the feature belongs to
    :param feature_name: name of the feature
    :param min_relative_frequency: the relative frequency with which a particular
    feature value has to occur within the sample, for it to be selected.
    :param limit_observations: how many observation-values to return at max.
    :param fallback_to_range_based: wether to fallback to a range based sampling
    approach, in case no single feature value is selected by frequency constraints
    :return: a 1D numpy array with the selected feature values
    """

    feature_sr = _select_feature(sample=sample, feature_name=feature_name)

    observed = _sample_by_frequency(
        series=feature_sr,
        min_relative_frequency=min_relative_frequency,
        limit_observations=limit_observations,
    )

    if len(observed) == 0 and fallback_to_range_based:
        return _sample_across_value_range(
            series=feature_sr, limit_observations=limit_observations
        )
    else:
        return observed


def observed_continuous_feature_values(
    sample: Sample,
    feature_name: str,
    limit_observations: int = DEFAULT_LIMIT_OBSERVATIONS,
) -> np.ndarray:
    """
    Get an array of observed values for a particular continuous feature

    :param sample: Sample the feature belongs to
    :param feature_name: name of the feature
    :param limit_observations: how many observation-values to return at max.
    :return: a 1D numpy array with the selected feature values
    """
    feature_sr = _select_feature(sample=sample, feature_name=feature_name)

    return _sample_across_value_range(
        series=feature_sr, limit_observations=limit_observations
    )


def _sample_by_frequency(
    series: pd.Series,
    min_relative_frequency: float = DEFAULT_MIN_RELATIVE_FREQUENCY,
    limit_observations: int = DEFAULT_LIMIT_OBSERVATIONS,
) -> np.ndarray:
    """
    Sample a series by relative frequency (value counts)
    :param series: pandas series to be sampled from
    :param min_relative_frequency: the relative frequency with which a particular
    feature value has to occur within the sample, for it to be selected.
    :param limit_observations: how many observation-values to return at max.
    :return: np.ndarray of value samples
    """

    # get value counts
    times_observed_sr = series.value_counts()

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
    return observed_filtered


def _sample_across_value_range(
    series: pd.Series, limit_observations: int
) -> np.ndarray:
    """
    Samples across a value range by picking evenly spread out indices of the
    unique series or (if under limit) simply returning all unique values.

    :param series: pandas series to be sampled from
    :param limit_observations: how many observation-values to return at max.
    :return: np.ndarray of value samples
    """
    unique_values_sorted: np.ndarray = series.copy().unique()
    unique_values_sorted.sort()
    # are there more unique-values than allowed by the passed limit?
    if len(unique_values_sorted) > limit_observations:
        # use np.linspace to spread out array indices evenly within bounds
        value_samples = np.linspace(
            0, len(unique_values_sorted) - 1, limit_observations
        ).astype(int)
        # return selected feature values out of all unique feature values
        return unique_values_sorted[value_samples]
    else:
        # return all unique values, since they are within limit bound
        return unique_values_sorted


def _select_feature(sample: Sample, feature_name: str) -> pd.Series:
    """

    :param sample:
    :param feature_name:
    :return:
    """
    # get the series of the feature and drop NAs
    # todo: should we <always> drop-na??
    return sample.features.loc[:, feature_name].dropna()
