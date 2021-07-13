"""
Implementation of FACET's :class:`.SampleBalancer` class.
"""

import logging
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState

from pytools.api import AllTracker

from ._sample import Sample

log = logging.getLogger(__name__)

__all__ = ["SampleBalancer"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class SampleBalancer:
    """
    Balances the target distribution of a :class:`.Sample` along class labels or bins,
    by either over- or under-sampling its observations.
    """

    #: constant value for ``bins`` parameter, designating unique values of the
    #: target variable as subsampling bins
    BINS_LABELS = "labels"

    bin_ratio: Union[float, pd.Series]

    def __init__(
        self,
        *,
        bin_ratio: Union[float, Mapping[Any, float], pd.Series],
        bins: Union[str, int],
        max_observations: int = None,
        random_state: Union[int, RandomState, None] = None,
    ) -> None:
        """
        :param bin_ratio: The desired target class ratio after balancing, either
            indicating the maximum ratio between the minority class and any other
            class as a positive scalar, or indicating the target ratio among multiple
            classes as a dictionary mapping class labels to scalars.
        :param bins: either ``%LABELS%`` to treat target variables as class labels,
            or an integer greater than 1 indicating the number of equally-sized ranges
            of target values to use as bins
        :param max_observations: upper limit for the resulting sample size (optional)
        :param random_state: optional random seed or random state for resampling
        """
        if bins is None:
            bins = SampleBalancer.BINS_LABELS
        elif not (
            bins == SampleBalancer.BINS_LABELS or isinstance(bins, int) and bins > 1
        ):
            raise ValueError(
                f"arg bins={bins!r} expected to be {SampleBalancer.BINS_LABELS!r}, "
                f"or an integer larger than 1"
            )

        if isinstance(bin_ratio, float):
            if bin_ratio < 1.0:
                raise ValueError(f"arg bin_ratio={bin_ratio} must not be less than 1.0")

        else:
            if isinstance(bin_ratio, Mapping):
                bin_ratio = pd.Series(data=bin_ratio)
            elif not isinstance(bin_ratio, pd.Series):
                raise TypeError(
                    f"unsupported type {type(bin_ratio).__name__} for arg bin_ratio"
                )

            if pd.api.types.is_numeric_dtype(bin_ratio):
                raise TypeError(
                    "arg bin_ratio must map to numeric values, got "
                    f"{bin_ratio.dtype} values instead"
                )

            if (bin_ratio <= 0.0).any():
                raise ValueError("arg max_ratio must be positive for all values")

            # normalize max ratios
            # noinspection PyArgumentList
            bin_ratio = bin_ratio / bin_ratio.sum()

        self.bin_ratio = bin_ratio
        self.bins = bins
        self.downsample = downsample
        self.max_observations = max_observations
        self.random_state = random_state

    __init__.__doc__ = __init__.__doc__.replace("%LABELS%", BINS_LABELS)

    def balance_weights(self, sample: Sample) -> Sample:
        """
        Balance the sample by balancing sample weights across classes.

        :param sample: the sample for which to balance weights
        :return: the balanced sample
        """
        pass

    def downsample(self, sample: Sample) -> Sample:
        """
        Balance the sample by undersampling over-represented classes.

        :param sample: the sample to downsample
        :return: the downsampled sample
        """
        pass

    def upsample(self, sample: Sample) -> Sample:
        """
        Balance the sample by oversampling under-represented classes.

        :param sample: the sample to upsample
        :return: the upsampled sample
        """
        pass

    def _balance(self, sample: Sample) -> Sample:
        """
        Balance the sample by over- or undersampling observations.

        :param sample: the sample to balance
        :return: the balanced sample
        """

        target = sample.target

        bin_sizes: pd.Series = target.value_counts(
            bins=None if self.bins == SampleBalancer.BINS_LABELS else self.bins,
            dropna=False,
        )

        # noinspection PyArgumentList
        bin_sizes_relative: pd.Series = bin_sizes / bin_sizes.sum()

        bin_ratio: Union[float, pd.Series] = self.bin_ratio
        downsample: bool = self.downsample
        max_observations = self.max_observations

        if isinstance(bin_ratio, pd.Series):
            # we normalize the counts by dividing by the maximum ratio
            normalized_counts = bin_sizes / bin_ratio

            # we determine the target number of counts by
            target_counts = bin_sizes.min() * bin_ratio
        else:
            # we have a single value for a maximum ratio between bin frequencies

            # determine the maximum bin count as a multiple of the least frequent bin
            bin_size_limit = (
                bin_sizes.min if downsample else bin_sizes.max
            )() * bin_ratio

            # we determine the new bin sizes that satisfy the lower/upper limit
            # determined from the bin_ratio
            bin_sizes_resampled = bin_sizes.where(
                lambda count: (
                    len(count) <= bin_size_limit
                    if downsample
                    else len(count) >= bin_size_limit
                ),
                other=bin_size_limit,
            )

            # we reduce bin sizes if we're above the total limit of observations
            bin_sizes_resampled_total = bin_sizes_resampled.sum()
            if bin_sizes_resampled_total > max_observations:
                bin_sizes_resampled = bin_sizes_resampled.floordiv(
                    max_observations
                ).astype(int)

            def _resample(values: pd.Series, target_size: int) -> pd.Series:
                actual_size = len(values)
                if actual_size == target_size:
                    return values
                elif actual_size < target_size:
                    return values.sample(n=target_size, replace=False)
                else:
                    return pd.concat(
                        values, values.sample(n=target_size - actual_size, replace=True)
                    )

            target_resampled: np.ndarray = np.vstack(
                _resample(values, bin_sizes_resampled.loc[label]).index.to_numpy()
                for label, values in target.groupby(target)
            )

            # … and based on this create the balanced subsample
            return sample.subsample(loc=target_resampled)
