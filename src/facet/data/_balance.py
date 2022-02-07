"""
Implementation of the sample balancer.
"""

from typing import Any, Dict, List

import pandas as pd

from pytools.api import AllTracker

from ._sample import Sample

__all__ = ["SampleBalancer"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())

_F_FACET_TARGET_PARTITION = "FACET_TARGET_PARTITION_"
_F_FACET_SAMPLE_WEIGHT = "FACET_SAMPLE_WEIGHT"


class SampleBalancer:
    """
    Balances the target distribution of a :class:`.Sample`, by over- or
    under-sampling its observations.
    Sample objects with multiple targets are not supported.
    """

    #
    # target_ratios_
    # partitioner

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
            scenario, all remaining frequency is allocated evenly across them. When
            frequencies for all classes are passed, they are expected to sum up to
            1.00 after rounding to two digits. Sampling with frequencies of 0 is not
            supported.
        :param oversample: Whether to use oversampling.
        :param undersample: Whether to use undersampling.
        """

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

        sampling_factors = self._find_sampling_factors(sample)

        # skip sampling if all factors 1 or nearly 1
        if all(round(sampling_factors, 6) == 1.0):
            return sample

        sampling_factors = sampling_factors.to_dict()

        if only_set_weights:
            weight_series = pd.Series(
                index=sample.target.index, name=_F_FACET_SAMPLE_WEIGHT
            )
            weight_series.loc[:] = 1.0

            for label, weight in sampling_factors.items():
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
                for label, factor in sampling_factors.items()
                if not (1.00000000 > factor > 1.000000001)
            }

            balanced_dfs = [
                observations[sample.target == label].sample(
                    frac=factor, replace=factor > 1.000000001
                )
                for label, factor in sampling_factors.items()
                if label in needs_sampling
            ]

            keep_as_is_dfs = [
                observations[sample.target == label]
                for label, factor in sampling_factors.items()
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

    def _find_sampling_factors(
        self,
        sample: Sample,
    ) -> pd.Series:
        class_count = sample.target.value_counts()
        target_frequencies = self._target_frequencies

        if not all(target_frequencies.index.isin(class_count.index)):
            unknowns = ",".join(
                target_frequencies.index[
                    ~target_frequencies.index.isin(class_count.index)
                ]
            )
            raise ValueError(
                f"arg target_frequencies specifies unknown class labels: {unknowns}"
            )

        cumulative_frequency = target_frequencies.sum()
        # check, if user specified frequencies for all found class labels in sample:
        if all(class_count.index.isin(target_frequencies.index)):
            if round(cumulative_frequency, 2) != 1.0:
                raise ValueError(
                    f"arg target_frequencies expects a cumulative frequency of 1.0, "
                    f"but is {cumulative_frequency}"
                )
        else:
            omitted_classes = set(
                class_count.index[~class_count.index.isin(target_frequencies.index)]
            )

            if round(cumulative_frequency, 2) == 1.0:
                raise ValueError(
                    "frequencies passed for arg target_frequencies specify cumulative "
                    f"frequency of 1.0, but class label(s) {','.join(omitted_classes)} "
                    "have been omitted and would end up with 0.0 – either specify them "
                    "directly or reduce allocation of frequencies to other classes."
                )
            else:
                remaining_frequency = 1 - cumulative_frequency
                target_frequencies = pd.concat(
                    [
                        target_frequencies,
                        pd.Series(
                            {
                                c: remaining_frequency / len(omitted_classes)
                                for c in omitted_classes
                            }
                        ),
                    ]
                )

        if self._oversample and self._undersample:
            sampling_factor = pd.Series(
                class_count.sum() * (target_frequencies / class_count),
                name="sample_factor",
            )

        elif self._oversample:
            relative_target_frequency = target_frequencies / target_frequencies.max()
            balanced_count = relative_target_frequency * class_count.max()
            preliminary_sample_factor = pd.Series(
                balanced_count / class_count, name="sample_factor"
            )

            # if any sample factor is < 1, scale up proportionally to keep oversampling
            if preliminary_sample_factor.min() < 1:
                sampling_factor = (
                    preliminary_sample_factor / preliminary_sample_factor.min()
                )
            else:
                sampling_factor = preliminary_sample_factor

        else:
            # perform only undersampling:
            relative_target_frequency = target_frequencies / target_frequencies.min()
            balanced_count = relative_target_frequency * class_count.min()
            preliminary_sample_factor = pd.Series(
                balanced_count / class_count, name="sample_factor"
            )

            # if any sample factor is > 1, scale down proportionally to keep
            # undersampling
            if preliminary_sample_factor.max() > 1:
                sampling_factor = (
                    preliminary_sample_factor / preliminary_sample_factor.max()
                )
            else:
                sampling_factor = preliminary_sample_factor

        return sampling_factor


__tracker.validate()
