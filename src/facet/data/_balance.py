from typing import Any, Dict, Iterable, Union

import numpy as np
import pandas as pd

from pytools.api import AllTracker

from facet.data import Sample

__all__ = ["SampleBalancer"]


#
# Ensure all symbols introduced below are included in __all__
#


__tracker = AllTracker(globals())

_F_FACET_TARGET_PARTITION = "FACET_TARGET_PARTITION_"


class SampleBalancer:
    """
    Balances the target distribution of a :class:`.Sample`, by either over- or
    under-sampling its observations.
    """

    def __init__(
        self,
        *,
        target_class_ratio: Union[float, Dict[Any, float]],
        bins: Union[str, int],
        undersample: bool = True,
    ) -> None:
        """
        :param target_class_ratio: The desired target class ratio after balancing,
            either indicating the maximum ratio between the minority class and any other
            class as a positive scalar in range ``]0,1]`` or indicating the target ratio
            of each minority class, as a dictionary mapping class labels to scalars.
        :param bins: either ``"labels"`` to treat target variables as class labels,
            or an integer greater than 1 indicating the number of equally-sized ranges
            of target values to use as bins
        :param undersample: boolean parameter, whether majority class should be
            undersampled, or minority class(es) should be oversampled
        """

        if isinstance(target_class_ratio, float):
            if not 0 < target_class_ratio <= 1:
                raise ValueError("'target_class_ratio' not in range ]0,1]")
        elif isinstance(target_class_ratio, Dict):
            if not bins == "labels":
                raise ValueError(
                    "If Dict passed for 'target_class_ratio', "
                    "then bins='labels' is required"
                )
        else:
            raise TypeError(
                f"Unsupported type '{type(target_class_ratio)}' for target_class_ratio"
            )

        if isinstance(bins, int):
            if bins <= 1:
                raise ValueError("'bins' needs to be >1")

            self._partition_target = True

        elif isinstance(bins, str) and bins == "labels":
            self._partition_target = False

        else:
            raise ValueError("'bins' needs to be an integer >1 or 'labels'")

        self._target_class_ratio = target_class_ratio
        self._bins = bins
        self._undersample = undersample

    def balance(self, sample: Sample) -> Sample:
        """
        Balance the sample by either oversampling observations of minority labels
        (partitions) or undersampling observations of majority labels (partitions)

        :param sample: the sample to balance
        :return: the balanced sample
        """
        # not imported globally, to avoid cross import error with Crossfit & Simulation
        from facet.simulation.partition import (
            CategoryPartitioner,
            ContinuousRangePartitioner,
            IntegerRangePartitioner,
            Partitioner,
        )

        partitioner: Partitioner

        if self._partition_target:
            target_dtype = sample.target.dtype

            if pd.api.types.is_integer_dtype(target_dtype):
                partitioner = IntegerRangePartitioner(max_partitions=self._bins)
            elif pd.api.types.is_float_dtype(target_dtype):
                partitioner = ContinuousRangePartitioner(max_partitions=self._bins)
            else:
                raise ValueError(
                    f"Sample target has unsupported dtype '{target_dtype}' "
                    f" in order to partition."
                )

            partitioner.fit(values=sample.target)
            partitioned_target_series = self._make_partition_series(sample, partitioner)

        else:
            partitioner = CategoryPartitioner()
            partitioner.fit(values=sample.target)
            partitioned_target_series = None

        max_freq_index = np.argmax(partitioner.frequencies_)
        max_freq_label = partitioner.partitions_[max_freq_index]
        max_freq = partitioner.frequencies_[max_freq_index]

        frequency_ratios = np.array(partitioner.frequencies_) / max_freq
        freq_ratio_by_label = {
            label: ratio
            for label, ratio in zip(partitioner.partitions_, frequency_ratios)
        }

        if self._undersample:
            if isinstance(self._target_class_ratio, float):
                # base undersample factor on class with smallest frequency ratio to
                # majority class, e.g. pick the maximum amount of undersampling
                # of the majority class, as needed to meet constraints:
                min_ratio = np.min(frequency_ratios)

                # return the input Sample as-is, if no undersampling needed:
                if min_ratio >= self._target_class_ratio:
                    return sample

                undersample_factor = min_ratio / self._target_class_ratio
            else:
                self._target_class_ratio: Dict[Any, float]
                self._check_label_dict(
                    target_class_ratio=self._target_class_ratio,
                    minority_labels={
                        label
                        for label in partitioner.partitions_
                        if label != max_freq_label
                    },
                )

                # return the input Sample as-is, if no undersampling needed:
                no_undersampling_needed = [
                    freq_ratio_by_label[label] >= target_class_ratio
                    for label, target_class_ratio in self._target_class_ratio.items()
                ]
                if all(no_undersampling_needed):
                    return sample

                undersampling_factors = [
                    freq_ratio_by_label[label] / target_class_ratio
                    for label, target_class_ratio in self._target_class_ratio.items()
                ]

                # pick the minimum factor, e.g. the maximum amount of undersampling
                # of the majority class, as needed to meet constraints:
                undersample_factor = min(undersampling_factors)

            # check undersample_factor
            # (exception should never occur as 'target_class_ratio' is sanitised):
            if undersample_factor >= 1:
                raise ValueError(
                    f"Calculated undersample factor is >=1 : {undersample_factor}"
                )

            return self._make_sample_with_majority_undersampled(
                sample=sample,
                undersample_factor=undersample_factor,
                majority_label=max_freq_label,
            )
        else:
            if isinstance(self._target_class_ratio, float):
                # if single target ratio passed, set it for all labels:
                target_class_ratio_by_label = {
                    label: self._target_class_ratio for label in partitioner.partitions_
                }
            else:
                target_class_ratio_by_label = self._target_class_ratio

            below_target_ratio = [
                (label, current_ratio)
                for label, current_ratio in zip(
                    partitioner.partitions_, frequency_ratios
                )
                if label != max_freq_label
                and current_ratio < target_class_ratio_by_label[label]
            ]

            # return the input Sample as-is, if no oversampling needed:
            if len(below_target_ratio) == 0:
                return sample

            oversampling_factors = {
                label: target_class_ratio_by_label[label] / current_ratio
                for label, current_ratio in below_target_ratio
            }

            # check oversampling_factors
            # (exception should never occur as 'target_class_ratio' is sanitised):
            if any([f for label, f in oversampling_factors.items() if f <= 1]):
                raise ValueError(
                    f"A calculated oversampling factor is <= 1 :{oversampling_factors}"
                )

            return self._make_sample_with_minority_oversampled(
                sample=sample,
                oversampling_factors=oversampling_factors,
                labels=partitioner.partitions_,
                partitioned_target_series=partitioned_target_series,
            )

    @staticmethod
    def _make_partition_series(sample: Sample, partitioner) -> pd.Series:
        """
        Add a pseudo-label series to a Sample with numerical target series.

        :param sample: the input Sample
        :param partitioner: a facet.simulation.partition.RangePartitioner, fit on
            sample
        :return: a Pandas series with partition center values as labels
        """
        from facet.simulation.partition import RangePartitioner

        partitioner: RangePartitioner

        partitioned_target_series = sample.target.copy()
        partitioned_target_series.name = _F_FACET_TARGET_PARTITION

        for partition_center, partition_bounds in zip(
            partitioner.partitions_, partitioner.partition_bounds_
        ):
            partitioned_target_series.loc[
                (partitioned_target_series >= partition_bounds[0])
                & (partitioned_target_series < partition_bounds[1])
            ] = partition_center

        return partitioned_target_series

    @staticmethod
    def _make_sample_with_majority_undersampled(
        sample: Sample,
        undersample_factor: float,
        majority_label: Any,
        partitioned_target_series: pd.Series = None,
    ) -> Sample:
        """
        Return a new Sample with observations of minority classes/partitions
        undersampled.

        :param sample: the input Sample
        :param undersample_factor: the factor of how much to undersample the majority
        :param majority_label: label/partition center denoting the majority
        :param partitioned_target_series: a pandas.Series with partition values
            corresponding to a numeric Sample.target series (optional)
        :return: the balanced Sample
        """
        if partitioned_target_series is None:
            target_series = sample.target
        else:
            target_series = partitioned_target_series

        keep_minority_idx = target_series != majority_label

        keep_majority_idx = (target_series == majority_label).sample(
            frac=undersample_factor
        )
        keep_index = keep_minority_idx | keep_majority_idx

        return sample.subsample(loc=keep_index)

    @staticmethod
    def _make_sample_with_minority_oversampled(
        sample: Sample,
        oversampling_factors: Dict[Any, float],
        labels: Iterable[Any],
        partitioned_target_series: pd.Series = None,
    ) -> Sample:
        """
        Return a new Sample with observations of minority classes/partitions
        oversampled.

        :param sample: the input Sample
        :param oversampling_factors: dictionary mapping oversampling factors to class
            labels/partition centers
        :param labels: iterable of all labels in Sample – including untouched ones
        :param partitioned_target_series: a pandas.Series with partition values
            corresponding to a numeric Sample.target series(optional)
        :return: the balanced Sample
        """

        labels_to_oversample = {label for label, _ in oversampling_factors.items()}

        if partitioned_target_series is None:
            target_series = sample.target
        else:
            target_series = partitioned_target_series

        observations: pd.DataFrame = pd.concat(
            [sample.features, sample.target, partitioned_target_series], axis=1
        )

        oversampled_dfs = [
            observations[target_series == label].sample(frac=factor, replace=True)
            for label, factor in oversampling_factors.items()
        ]

        keep_as_is_dfs = [
            observations[target_series == label]
            for label in labels
            if label not in labels_to_oversample
        ]

        new_observations = oversampled_dfs + keep_as_is_dfs
        new_observations_df = pd.concat(new_observations, axis=0)

        if partitioned_target_series is not None:
            new_observations_df = new_observations_df.drop(
                columns=[_F_FACET_TARGET_PARTITION]
            )

        return Sample(
            observations=new_observations_df,
            target_name=sample.target_name,
        )

    @staticmethod
    def _check_label_dict(
        target_class_ratio: Dict[Any, float], minority_labels: Iterable[Any]
    ) -> None:
        """
        Check a passed dictionary with target ratios for class labels.

        :param target_class_ratio: user provided mapping of label names to ratio
        :param minority_labels: minority labels found in the input Sample
        """
        if not set(target_class_ratio.keys()) == set(minority_labels):
            raise ValueError(
                "Keys in 'target_class_ratio' dict do not "
                "match with minority class labels"
            )


__tracker.validate()
