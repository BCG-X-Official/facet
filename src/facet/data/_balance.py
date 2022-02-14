"""
Implementation of the sample balancer.
"""
import abc
from abc import ABCMeta
from typing import Any, Dict, List

import pandas as pd

from pytools.api import AllTracker

from ._sample import Sample

__all__ = [
    "SampleBalancer",
    "UniformSampleBalancer",
    "TargetFrequencySampleBalancer",
]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())

_F_FACET_TARGET_PARTITION = "FACET_TARGET_PARTITION_"
_F_FACET_SAMPLE_WEIGHT = "FACET_SAMPLE_WEIGHT"


class SampleBalancer(metaclass=ABCMeta):
    """
    Balances the target distribution of a :class:`.Sample`, by over- or
    under-sampling its observations.
    Sample objects with multiple targets are not supported.
    """

    _target_sampling_frequencies: pd.Series
    _sampling_factors: pd.Series
    _class_count: pd.Series

    def __init__(
        self,
        *,
        oversample: bool = True,
        undersample: bool = True,
    ) -> None:
        """
        :param oversample: Whether to use oversampling.
        :param undersample: Whether to use undersampling.
        """

        if not oversample and not undersample:
            raise ValueError(
                "args oversample and undersample are both false. enable at least one"
            )

        self._oversample = oversample
        self._undersample = undersample

    def _balance(self, sample: Sample, only_set_weights: bool) -> Sample:
        """
        Balance the sample by either oversampling observations of minority labels
        (partitions) or undersampling observations of majority labels (partitions)

        :param sample: the sample to balance
        :param only_set_weights: do not touch observations of sample, instead add a new
            series with sample weights for each observation
        :return: the balanced sample
        """
        if isinstance(sample.target_name, List):
            raise ValueError(
                "sample to balance should be single target, but has multiple"
            )

        self._set_class_count(sample)
        self._set_target_sampling_frequencies()
        self._set_sampling_factors()

        # skip sampling if all factors 1 or nearly 1
        if all(round(self._sampling_factors, 6) == 1.0):
            return sample

        if only_set_weights:
            weight_series = pd.Series(
                index=sample.target.index, name=_F_FACET_SAMPLE_WEIGHT
            )
            weight_series.loc[:] = 1.0

            for label, weight in self._sampling_factors.iteritems():
                weight_series[sample.target == label] = weight

            observations: pd.DataFrame = pd.concat(
                [sample.features, weight_series, sample.target], axis=1
            )

            return Sample(
                observations=observations,
                target_name=sample.target_name,
                weight_name=_F_FACET_SAMPLE_WEIGHT,
            )
        else:
            observations: pd.DataFrame = pd.concat(
                [sample.features, sample.target], axis=1
            )
            needs_sampling = {
                label
                for label, factor in self._sampling_factors.iteritems()
                if not round(factor, 6) == 1.0
            }

            balanced_dfs = [
                observations[sample.target == label].sample(
                    frac=factor, replace=factor > 1.0
                )
                for label, factor in self._sampling_factors.iteritems()
                if label in needs_sampling
            ]

            keep_as_is_dfs = [
                observations[sample.target == label]
                for label, factor in self._sampling_factors.iteritems()
                if label not in needs_sampling
            ]

            new_observations = balanced_dfs + keep_as_is_dfs
            new_observations_df = pd.concat(new_observations, axis=0)

            return Sample(
                observations=new_observations_df,
                target_name=sample.target_name,
            )

    def balance(self, sample: Sample) -> Sample:
        """
        Balance the sample by over- and/or undersampling.

        :param sample: the sample to balance
        :return: the balanced sample
        """
        return self._balance(sample, False)

    def set_balanced_weights(self, sample: Sample) -> Sample:
        """
        Enhance the sample by adding a new series which captures the sample weight
        of each observation. Weights are derived from sampling factors to balance
        observation as specified in target_frequencies.

        :param sample: the sample to balance
        :return: the sample with an additional series with weights
        """
        if sample.weight is not None:
            raise ValueError(
                f"Sample has existing weight series named {sample.weight_name} "
                f"which would be overridden."
            )

        return self._balance(sample, True)

    @abc.abstractmethod
    def _set_target_sampling_frequencies(self) -> None:
        pass

    def _set_class_count(self, sample: Sample) -> None:
        self._class_count = sample.target.value_counts()

    def _set_sampling_factors(self) -> None:

        if self._oversample and self._undersample:
            sampling_factors = pd.Series(
                self._class_count.sum()
                * (self._target_sampling_frequencies / self._class_count),
                name="sample_factor",
            )

        elif self._oversample:
            relative_target_frequency = (
                self._target_sampling_frequencies
                / self._target_sampling_frequencies.max()
            )
            balanced_count = relative_target_frequency * self._class_count.max()
            preliminary_sample_factor = pd.Series(
                balanced_count / self._class_count, name="sample_factor"
            )

            # if any sample factor is < 1, scale up proportionally to keep oversampling
            if preliminary_sample_factor.min() < 1:
                sampling_factors = (
                    preliminary_sample_factor / preliminary_sample_factor.min()
                )
            else:
                sampling_factors = preliminary_sample_factor

        else:
            # perform only undersampling:
            relative_target_frequency = (
                self._target_sampling_frequencies
                / self._target_sampling_frequencies.min()
            )
            balanced_count = relative_target_frequency * self._class_count.min()
            preliminary_sample_factor = pd.Series(
                balanced_count / self._class_count, name="sample_factor"
            )

            # if any sample factor is > 1, scale down proportionally to keep
            # undersampling
            if preliminary_sample_factor.max() > 1:
                sampling_factors = (
                    preliminary_sample_factor / preliminary_sample_factor.max()
                )
            else:
                sampling_factors = preliminary_sample_factor

        # if undersampling was performed – alone or in combination with oversampling -
        # then check if absolute frequency of any class would drop below 1:
        if self._undersample:
            resulting_count: pd.Series = round(self._class_count * sampling_factors, 0)
            zero_obs_sampled = resulting_count[resulting_count == 0.0]
            if len(zero_obs_sampled) > 0:
                raise ValueError(
                    f"Undersampling of {','.join(zero_obs_sampled.index)} to "
                    f"meet target frequencies leads to 0 retained observations. "
                    f"consider to allow (only) oversampling to avoid this."
                )

        self._sampling_factors = sampling_factors


class UniformSampleBalancer(SampleBalancer):
    """
    TBD
    """

    def __init__(
        self,
        *,
        balance_pct: float = 0.5,
        oversample: bool = True,
        undersample: bool = True,
    ) -> None:
        """
        :param balance_pct: Share of observations to be uniformly rebalanced. Expected
            in range ]0.0,1.0].
        :param oversample: Whether to use oversampling.
        :param undersample: Whether to use undersampling.
        """
        super().__init__(oversample=oversample, undersample=undersample)

        if not oversample and not undersample:
            raise ValueError(
                "args oversample and undersample are both false. enable at least one"
            )

        if isinstance(balance_pct, float):
            if not 0 < balance_pct <= 1:
                raise ValueError(
                    f"arg rebalance_pct expected in range ]0,1]. "
                    f"but is: {balance_pct}"
                )
        else:
            raise TypeError(
                f"arg rebalance_pct should be float "
                f"but is a {type(balance_pct).__name__}"
            )

        self._oversample = oversample
        self._undersample = undersample
        self._balance_pct = balance_pct

    def _set_target_sampling_frequencies(self) -> None:
        uniform_balanced_pct_per_class = self._balance_pct / len(self._class_count)
        remaining_frequency = 1.0 - self._balance_pct
        allocated_of_remaining = (
            self._class_count / self._class_count.sum()
        ) * remaining_frequency
        self._target_sampling_frequencies = (
            uniform_balanced_pct_per_class + allocated_of_remaining
        )


class TargetFrequencySampleBalancer(SampleBalancer):
    """
    tbd
    """

    def __init__(
        self,
        *,
        target_frequencies: Dict[Any, float],
        oversample: bool = True,
        undersample: bool = True,
    ) -> None:
        """
        :param target_frequencies: Dictionary assigning desired target frequencies to
            class labels. Frequency values are expected as float within the ]0,1[
            interval. When omitting frequency for a class label, it is assigned the
            unallocated frequency: i.e. in binary classification one can specify just a
            single class label to balance with frequency 0.3, then 0.7 is used for the
            second class. When omitting multiple classes in a multiclass classification
            scenario, all remaining frequency is allocated across them, weighted on
            relative frequency ratios in the original sample. When frequencies for all
            classes are passed, they are expected to sum up to 1.00 after rounding to
            two digits. Sampling with frequencies of 0 is not supported.
        :param oversample: Whether to use oversampling.
        :param undersample: Whether to use undersampling.
        """
        super().__init__(oversample=oversample, undersample=undersample)

        if not oversample and not undersample:
            raise ValueError(
                "args oversample and undersample are both false. enable at least one"
            )

        if target_frequencies == {}:
            raise ValueError("arg target_frequencies is empty")

        for key, val in target_frequencies.items():
            if isinstance(val, float):
                if not 0 < val < 1:
                    raise ValueError(
                        f"target frequency for label {key} expected in range ]0,1[. "
                        f"but is: {val}"
                    )
            else:
                raise TypeError(
                    f"target frequency value for label {key} should be float "
                    f"but is a {type(val).__name__}"
                )

        self._oversample = oversample
        self._undersample = undersample
        self._target_frequencies = pd.Series(data=target_frequencies)

    def _set_target_sampling_frequencies(self) -> None:
        if not all(self._target_frequencies.index.isin(self._class_count.index)):
            unknowns = ",".join(
                self._target_frequencies.index[
                    ~self._target_frequencies.index.isin(self._class_count.index)
                ]
            )
            raise ValueError(
                f"arg target_frequencies specifies unknown class labels: {unknowns}"
            )

        cumulative_frequency = self._target_frequencies.sum()
        # check, if user specified frequencies for all found class labels in sample:
        if all(self._class_count.index.isin(self._target_frequencies.index)):
            if round(cumulative_frequency, 2) != 1.0:
                raise ValueError(
                    f"arg target_frequencies expects a cumulative frequency of 1.0, "
                    f"but is {cumulative_frequency}"
                )
            self._target_sampling_frequencies = self._target_frequencies
        else:
            omitted_classes = self._class_count.index[
                ~self._class_count.index.isin(self._target_frequencies.index)
            ]

            if round(cumulative_frequency, 2) == 1.0:
                raise ValueError(
                    "frequencies passed for arg target_frequencies specify cumulative "
                    f"frequency of 1.0, but class label(s) {','.join(omitted_classes)} "
                    "have been omitted and would end up with 0.0 – either specify them "
                    "directly or reduce allocation of frequencies to other classes."
                )
            else:
                remaining_frequency = 1 - cumulative_frequency
                # allocate remaining frequency based on relative frequencies among
                # omitted classes
                omitted_classes_weights = (
                    self._class_count[omitted_classes]
                    / self._class_count[omitted_classes].sum()
                )

                self._target_sampling_frequencies = pd.concat(
                    [
                        self._target_frequencies,
                        remaining_frequency * omitted_classes_weights,
                    ]
                )


__tracker.validate()
