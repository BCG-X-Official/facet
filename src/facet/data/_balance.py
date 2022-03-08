"""
Implementation of the sample balancer.
"""
import abc
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
    under sampling its observations.
    Sample objects with multiple targets are not supported.
    """

    _target_sampling_frequencies: pd.Series
    _sampling_factors: pd.Series
    _class_counts: pd.Series

    def __init__(
        self,
        *,
        oversample: bool = True,
        undersample: bool = True,
        random_state: Optional[
            Union[int, np.random.Generator, np.random.BitGenerator]
        ] = None,
    ) -> None:
        """
        :param oversample: Whether to use oversampling.
        :param undersample: Whether to use undersampling.
        :param random_state: Optional random seed. If provided balancing will be
            deterministic and `balance` call idempotent.
        """

        if not oversample and not undersample:
            raise ValueError(
                "args oversample and undersample are both false. enable at least one"
            )

        self._oversample = oversample
        self._undersample = undersample
        self.random_state = random_state

    def _balance(self, sample: Sample, only_set_weights: bool) -> Sample:
        """
        Balance the sample by oversampling observations of classes below the
        desired target frequencies and/or undersampling observations of classes
        above their desired target frequencies.

        :param sample: the sample to balance
        :param only_set_weights: do not touch observations of sample, instead add a new
            series with sample weights for each observation
        :return: the balanced sample
        """
        if isinstance(sample.target_name, List):
            raise ValueError(
                "sample to balance should be single target, but has multiple"
            )

        self._class_counts = sample.target.value_counts()
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

            observations = pd.concat(
                [sample.features, weight_series, sample.target], axis=1
            )

            return Sample(
                observations=observations,
                target_name=sample.target_name,
                weight_name=_F_FACET_SAMPLE_WEIGHT,
            )
        else:
            observation_idx = pd.Series(np.arange(len(sample)))

            rng = np.random.default_rng(self.random_state)

            def _sample(_observation_idx: pd.Series, factor: float) -> pd.DataFrame:
                n_observations = len(_observation_idx)
                n_target = int(n_observations * factor)

                idx_target = np.empty(n_target, dtype=int)

                start = 0
                if n_target > n_observations:
                    idx_full = np.arange(n_observations)
                    for i in range(n_target // n_observations):
                        idx_target[start : start + n_observations] = idx_full
                        start += n_observations

                idx_target[start:] = rng.choice(
                    n_observations, size=n_target % n_observations, replace=False
                )

                return _observation_idx.iloc[idx_target]

            observations_by_label: Dict[Any, pd.Series] = dict(
                tuple(observation_idx.groupby(sample.target.values))
            )

            new_observation_idx_sr = pd.concat(
                [
                    _sample(observations_by_label[label], factor)
                    if not round(factor, 6) == 1.0
                    else observation_idx[sample.target == label]
                    for label, factor in (self._sampling_factors.iteritems())
                ],
                axis=0,
            )

        return sample.subsample(iloc=new_observation_idx_sr)

    def balance(self, sample: Sample) -> Sample:
        """
        Balance the sample by oversampling observations of classes below the
        desired target frequencies and/or undersampling observations of classes
        above their desired target frequencies.

        :param sample: the sample to balance
        :return: the balanced sample
        """
        return self._balance(sample, False)

    def set_balanced_weights(self, sample: Sample) -> Sample:
        """
        Enhance the sample by adding a new series which captures the sample weight
        of each observation. Weights are derived from sampling factors to balance
        observations using over- and/or undersampling of classes.

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

    def _set_sampling_factors(self) -> None:

        if self._oversample and self._undersample:
            sampling_factors = pd.Series(
                self._class_counts.sum()
                * (self._target_sampling_frequencies / self._class_counts),
                name="sample_factor",
            )

        elif self._oversample:
            relative_target_frequency = (
                self._target_sampling_frequencies
                / self._target_sampling_frequencies.max()
            )
            balanced_count = relative_target_frequency * self._class_counts.max()
            preliminary_sample_factor = pd.Series(
                balanced_count / self._class_counts, name="sample_factor"
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
            balanced_count = relative_target_frequency * self._class_counts.min()
            preliminary_sample_factor = pd.Series(
                balanced_count / self._class_counts, name="sample_factor"
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
            resulting_count: pd.Series = round(self._class_counts * sampling_factors, 0)
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
    Balances the target distribution of a :class:`.Sample`, by over- or
    under sampling its observations.

    The :class:`.UniformSampleBalancer` is instantiated with ``balance_pct`` indicating
    the share of observations to uniformly rebalance. If passing ``1.0``, this leads to
    a Sample being balanced in a way where each class is represented with the same
    amount of observations. E.g. in a binary target Sample, each class occurs at a
    relative frequency of 0.5, in a Sample with four targets, each class occurs at a
    frequency of 0.25, and so on.
    If passing ``balance_pct < 1.0``, uniform rebalancing is only applied to this given
    share of the Sample. The remaining share is allocated across classes matching their
    relative frequencies within the original Sample.
    E.g. when passing ``balance_pct = 0.5``, then 50% of the sampling is uniform,
    and the remaining 50% corresponds to the original class frequencies.

    Sample objects with multiple targets are not supported.
    """

    def __init__(
        self,
        *,
        balance_pct: float = 0.5,
        oversample: bool = True,
        undersample: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """
        :param balance_pct: Share of observations to be uniformly rebalanced. Expected
            in range ]0,1]. If a value below 1.0 is passed, the remaining share
            of observations to resample is allocated to each class using its
            corresponding relative frequency within the original Sample.
        :param oversample: Whether to use oversampling.
        :param undersample: Whether to use undersampling.
        :param random_state: Optional random seed. If provided balancing will be
            deterministic and `balance` call idempotent.
        """
        super().__init__(
            oversample=oversample, undersample=undersample, random_state=random_state
        )

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
        uniform_balanced_pct_per_class = self._balance_pct / len(self._class_counts)
        remaining_frequency = 1.0 - self._balance_pct
        allocated_of_remaining = (
            self._class_counts / self._class_counts.sum()
        ) * remaining_frequency
        self._target_sampling_frequencies = (
            uniform_balanced_pct_per_class + allocated_of_remaining
        )


class TargetFrequencySampleBalancer(SampleBalancer):
    """
    Balances the target distribution of a :class:`.Sample`, by over- or
    under sampling its observations.

    The :class:`.TargetFrequencySampleBalancer` is instantiated with argument
    ``target_frequencies`` indicating the desired target frequency per class after
    balancing.

    Sample objects with multiple targets are not supported.
    """

    def __init__(
        self,
        *,
        target_frequencies: Dict[Any, float],
        oversample: bool = True,
        undersample: bool = True,
        random_state: Optional[int] = None,
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
        :param random_state: Optional random seed. If provided balancing will be
            deterministic and `balance` call idempotent.
        """
        super().__init__(
            oversample=oversample, undersample=undersample, random_state=random_state
        )

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
        if not all(self._target_frequencies.index.isin(self._class_counts.index)):
            unknowns = ",".join(
                self._target_frequencies.index[
                    ~self._target_frequencies.index.isin(self._class_counts.index)
                ]
            )
            raise ValueError(
                f"arg target_frequencies specifies unknown class labels: {unknowns}"
            )

        cumulative_frequency = self._target_frequencies.sum()
        # check, if user has specified frequencies for all found class labels in sample:
        if all(self._class_counts.index.isin(self._target_frequencies.index)):
            if round(cumulative_frequency, 2) != 1.0:
                raise ValueError(
                    f"arg target_frequencies expects a cumulative frequency of 1.0, "
                    f"but is {cumulative_frequency}"
                )
            self._target_sampling_frequencies = self._target_frequencies
        else:
            omitted_classes = self._class_counts.index[
                ~self._class_counts.index.isin(self._target_frequencies.index)
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
                    self._class_counts[omitted_classes]
                    / self._class_counts[omitted_classes].sum()
                )

                self._target_sampling_frequencies = pd.concat(
                    [
                        self._target_frequencies,
                        remaining_frequency * omitted_classes_weights,
                    ]
                )


__tracker.validate()
