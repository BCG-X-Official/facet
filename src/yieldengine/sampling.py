import math
from abc import ABC, ABCMeta, abstractmethod
from typing import *

import numpy as np
import pandas as pd

from yieldengine import Sample
from yieldengine.df import ListLike

DEFAULT_MAX_PARTITIONS = 20

DEFAULT_MIN_RELATIVE_FREQUENCY = 0.05
DEFAULT_LIMIT_OBSERVATIONS = 20

ValueType = TypeVar("ValueType")
NumericType = TypeVar("NumericType", bound=Union[int, float])
CategoryType = TypeVar("CategoryType", bound=Union[str, object])


class ValuePartitioning(ABC, Generic[ValueType]):
    """
    A partitioning of a set of observed values, for use in visualizations and
    "virtual experiment" simulations
    """

    @abstractmethod
    def partitions(self) -> Iterable[ValueType]:
        """
        :return: for each partition, return a central value representing the partition
        """
        pass

    @abstractmethod
    def frequencies(self) -> Iterable[int]:
        """
        :return: for each partition, the number of observed values that fall within
        the partition
        """

    @property
    @abstractmethod
    def n_partitions(self) -> int:
        pass


class RangePartitioning(
    ValuePartitioning[NumericType], Generic[NumericType], metaclass=ABCMeta
):
    def __init__(
        self,
        values: ListLike[NumericType],
        max_partitions: int = DEFAULT_MAX_PARTITIONS,
        lower_bound: Optional[NumericType] = None,
        upper_bound: Optional[NumericType] = None,
    ) -> None:
        super().__init__()

        if lower_bound is None:
            lower_bound = np.min(values)
        if upper_bound is None:
            upper_bound = np.max(values)
        if upper_bound <= lower_bound:
            raise ValueError("arg lower_bound >= arg upper_bound")

        # calculate the step count based on the maximum number of partitions,
        # rounded to the next-largest rounded value ending in 1, 2, or 5
        self._step = step = self._step_size(lower_bound, upper_bound, max_partitions)

        # calculate centre values of the first and last partition;
        # both are rounded to multiples of the step size
        self._first_partition = first_partition = (
            math.floor((lower_bound + step / 2) / step) * step
        )
        self._last_partition = math.ceil((upper_bound - step / 2) / step) * step
        self._n_partitions = n_partitions = (
            int(round((self._last_partition - self._first_partition) / self._step)) + 1
        )

        def _frequencies() -> List[int]:
            partition_indices = [
                int(round(value - first_partition) / step) for value in values
            ]
            frequencies = [0] * n_partitions
            for idx in partition_indices:
                if 0 <= idx < n_partitions:
                    frequencies[idx] += 1

            return frequencies

        self._frequencies = _frequencies()

    def partitions(self) -> Sequence[NumericType]:
        return [idx * self._step for idx in range(0, self.n_partitions)]

    def frequencies(self) -> Iterable[int]:
        return self._frequencies

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

    @property
    def partition_width(self) -> NumericType:
        return self._step

    @staticmethod
    @abstractmethod
    def _step_size(
        lower_bound: NumericType, upper_bound: NumericType, max_partitions: int
    ) -> NumericType:
        pass


class ContinuousValuePartitioning(RangePartitioning[float]):
    @staticmethod
    def _step_size(
        lower_bound: float, upper_bound: float, max_partitions: int
    ) -> float:
        return ceil_step((upper_bound - lower_bound) / (max_partitions - 1))


class DiscreteValuePartitioning(RangePartitioning[int]):
    @staticmethod
    def _step_size(lower_bound: int, upper_bound: int, max_partitions: int) -> int:
        return max(
            1, int(ceil_step((upper_bound - lower_bound) / (max_partitions - 1)))
        )


class CategoricalPartitioning(ValuePartitioning[CategoryType]):
    def __init__(
        self,
        values: ListLike[CategoryType],
        max_partitions: int = DEFAULT_MAX_PARTITIONS,
    ) -> None:
        super().__init__()

        value_counts = pd.Series(data=values).value_counts(ascending=False)
        self._frequencies = value_counts.values[:max_partitions]
        self._partitions = value_counts.index.values[:max_partitions]

    def partitions(self) -> Iterable[CategoryType]:
        return self._partitions

    def frequencies(self) -> Iterable[int]:
        return self._frequencies

    @property
    def n_partitions(self) -> int:
        return len(self._partitions)


def observed_categorical_feature_valuxes(
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


def ceil_step(step: float):
    """
    :param step: the step size to round by
    :return: the nearest step size in the series
             (..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, ...)
    """
    if step <= 0:
        raise ValueError("arg step must be positive")

    return min(10 ** math.ceil(math.log10(step * m)) / m for m in [1, 2, 5])
