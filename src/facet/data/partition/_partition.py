"""
Core implementation of :mod:`facet.simulation.partition`
"""

import logging
import math
import operator as op
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Any, Generic, Iterable, Optional, Sequence, Tuple, TypeVar

import numpy as np

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin

log = logging.getLogger(__name__)

__all__ = [
    "Partitioner",
    "RangePartitioner",
    "ContinuousRangePartitioner",
    "IntegerRangePartitioner",
    "CategoryPartitioner",
]


#
# Type variables
#


T_Self = TypeVar("T_Self")
T_Values = TypeVar("T_Values")
T_Values_Numeric = TypeVar("T_Values_Numeric", bound=Number)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class Partitioner(
    FittableMixin[Iterable[T_Values]], Generic[T_Values], metaclass=ABCMeta
):
    """
    Abstract base class of all partitioners.
    """

    DEFAULT_MAX_PARTITIONS = 20

    #: The central values for all partitions.
    _partitions: Optional[np.ndarray]

    #: The value counts for all partitions.
    _frequencies: Optional[np.ndarray]

    def __init__(self, max_partitions: Optional[int] = None) -> None:
        """
        :param max_partitions: the maximum number of partitions to generate; must
            be at least 2 (default: {DEFAULT_MAX_PARTITIONS})
        """
        if max_partitions is None:
            self._max_partitions = Partitioner.DEFAULT_MAX_PARTITIONS
        elif max_partitions < 2:
            raise ValueError(f"arg max_partitions={max_partitions} must be at least 2")
        else:
            self._max_partitions = max_partitions

        self._partitions = None
        self._frequencies = None

    __init__.__doc__ = __init__.__doc__.replace(
        "{DEFAULT_MAX_PARTITIONS}", repr(DEFAULT_MAX_PARTITIONS)
    )

    @property
    def max_partitions(self) -> int:
        """
        The maximum number of partitions to be generated by this partitioner.
        """
        return self._max_partitions

    @property
    def partitions_(self) -> np.ndarray:
        """
        The central values of all partitions.
        """
        self._ensure_fitted()
        return self._partitions

    @property
    def frequencies_(self) -> np.ndarray:
        """
        The count of data points in each partition.
        """
        self._ensure_fitted()
        return self._frequencies

    @property
    @abstractmethod
    def is_categorical(self) -> bool:
        """
        ``True`` if this is partitioner handles categorical values, ``False`` otherwise.
        """

    @abstractmethod
    def fit(self: T_Self, values: Iterable[T_Values], **fit_params: Any) -> T_Self:
        """
        Calculate the partitioning for the given observed values.

        :param values: a sequence of observed values as the empirical basis for
            calculating the partitions
        :param fit_params: optional fitting parameters
        :return: ``self``
        """

    @staticmethod
    def _as_non_empty_array(values: Iterable[Any]) -> np.ndarray:
        # ensure arg values is a non-empty array
        values = np.asarray(values)
        if len(values) == 0:
            raise ValueError("arg values is empty")
        return values


@inheritdoc(match="""[see superclass]""")
class RangePartitioner(
    Partitioner[T_Values_Numeric], Generic[T_Values_Numeric], metaclass=ABCMeta
):
    """
    Abstract base class of partitioners for numerical ranges.
    """

    def __init__(self, max_partitions: Optional[int] = None) -> None:
        """[see superclass]"""

        super().__init__(max_partitions)

        self._step: Optional[T_Values_Numeric] = None
        self._partition_bounds: Optional[
            Sequence[Tuple[T_Values_Numeric, T_Values_Numeric]]
        ] = None

    @property
    def lower_bound(self) -> T_Values_Numeric:
        """
        The lower bound of the partitioning.

        ``Null`` if no explicit lower bound is set.
        """
        return self._lower_bound

    @property
    def upper_bound(self) -> T_Values_Numeric:
        """
        The upper bound of the partitioning.

        ``Null`` if no explicit upper bound is set.
        """
        return self._upper_bound

    @property
    def is_categorical(self) -> bool:
        """
        ``False``
        """
        return False

    @property
    def partition_bounds_(self) -> Sequence[Tuple[T_Values_Numeric, T_Values_Numeric]]:
        """
        Return the endpoints of the intervals that delineate each partitions.

        :return: sequence of tuples (x, y) for every partition, where x is the
          inclusive lower bound of a partition range, and y is the exclusive upper
          bound of a partition range
        """
        self._ensure_fitted()
        return self._partition_bounds

    @property
    def partition_width_(self) -> T_Values_Numeric:
        """
        The width of each partition.
        """
        self._ensure_fitted()
        return self._step

    # noinspection PyMissingOrEmptyDocstring,PyIncorrectDocstring
    def fit(
        self: T_Self,
        values: Iterable[T_Values_Numeric],
        *,
        lower_bound: Optional[T_Values_Numeric] = None,
        upper_bound: Optional[T_Values_Numeric] = None,
        **fit_params: Any,
    ) -> T_Self:
        r"""
        Calculate the partitioning for the given observed values.

        The lower and upper bounds of the range to be partitioned can be provided
        as optional arguments.
        If no bounds are provided, the partitioner automatically chooses the lower
        and upper outlier thresholds based on the Tukey test, i.e.,
        :math:`[- 1.5 * \mathit{iqr}, 1.5 * \mathit{iqr}]`
        where :math:`\mathit{iqr}` is the inter-quartile range.

        :param values: a sequence of observed values as the empirical basis for
            calculating the partitions
        :param lower_bound: the inclusive lower bound of the elements to partition
        :param upper_bound: the inclusive upper bound of the elements to partition
        :param fit_params: optional fitting parameters
        :return: ``self``
        """

        self: RangePartitioner  # support type hinting in PyCharm

        values = self._as_non_empty_array(values)

        if lower_bound is None or upper_bound is None:
            q3q1 = np.nanquantile(values, q=[0.75, 0.25])
            inlier_range = op.sub(*q3q1) * 1.5  # iqr * 1.5

            if lower_bound is None:
                lower_bound = values[values >= q3q1[1] - inlier_range].min()

            if upper_bound is None:
                upper_bound = values[values <= q3q1[0] + inlier_range].max()

            if lower_bound == upper_bound:
                raise ValueError(
                    "insufficient variance in values; cannot infer partitioning bounds"
                )

        elif lower_bound >= upper_bound:
            raise ValueError(
                "arg lower_bound must be lower than arg upper_bound "
                f"but got: [{lower_bound}, {upper_bound})"
            )

        assert lower_bound < upper_bound

        # calculate the step count based on the maximum number of partitions,
        # rounded to the next-largest rounded value ending in 1, 2, or 5
        self._step = step = self._step_size(lower_bound, upper_bound)

        # calculate centre values of the first and last partition;
        # both are rounded to multiples of the step size
        first_partition = math.floor((lower_bound + step / 2) / step) * step
        last_partition = math.ceil((upper_bound - step / 2) / step) * step
        n_partitions = int(round((last_partition - first_partition) / self._step)) + 1

        self._partitions = partitions = np.round(
            first_partition + np.arange(n_partitions) * self._step,
            # round to the nearest power of 10 of the step variable
            int(-np.floor(np.log10(self._step))),
        ).tolist()

        center_offset_left = self._partition_center_offset
        center_offset_right = self._step - center_offset_left
        self._partition_bounds = [
            (center - center_offset_left, center + center_offset_right)
            for center in partitions
        ]

        # calculate the number of elements in each partitions

        # create the bins, starting with the lower bound of the first partition
        partition_bins = (first_partition - step / 2) + (
            step * np.arange(n_partitions + 1)
        )
        partition_indices = np.digitize(values, bins=partition_bins)

        # frequency counts will include left and right outliers, hence n_partitions + 2
        # and we exclude the first and last element of the result
        frequencies = np.bincount(partition_indices, minlength=n_partitions + 2)[1:-1]

        self._frequencies = frequencies

        return self

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._frequencies is not None

    @staticmethod
    def _ceil_step(step: float):
        """
        Round the step size (arbitrary float) to a human-readable number like 0.5, 1, 2.

        :param step: the step size to round by
        :return: the nearest greater or equal step size in the series
                 (..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, ...)
        """
        if step <= 0:
            raise ValueError("arg step must be positive")

        return min(10 ** math.ceil(math.log10(step * m)) / m for m in [1, 2, 5])

    @staticmethod
    @abstractmethod
    def _step_size(
        lower_bound: T_Values_Numeric, upper_bound: T_Values_Numeric
    ) -> T_Values_Numeric:
        # Compute the step size (interval length) used in the partitions
        pass

    @property
    @abstractmethod
    def _partition_center_offset(self) -> T_Values_Numeric:
        # Offset between center and endpoints of an interval
        pass


class ContinuousRangePartitioner(RangePartitioner[float]):
    """
    Partition numerical values in adjacent intervals of the same length.

    The range of intervals and interval size is computed based on attributes
    :attr:`.max_partitions`, :attr:`.lower_bound`, and :attr:`.upper_bound`.

    Partition boundaries and interval sized are chosen with interpretability in mind and
    are always a power of 10, or a multiple of 2 or 5 of a power of 10, e.g.
    0.1, 0.2, 0.5, 1.0, 2.0, 5.0, and so on.

    The intervals also satisfy the following conditions:

    - :attr:`lower_bound` is within the first interval
    - :attr:`upper_bound` is within the last interval

    For example, with :attr:`.max_partitions` = 10, :attr:`.lower_bound` = 3.3, and
    :attr:`.upper_bound` = 4.7, the resulting partitioning would be:
    [3.2, 3.4), [3.4, 3.6), [3.6, 3.8), [4.0, 4.2), [4.4, 4.6), [4.6, 4.8]
    """

    def _step_size(self, lower_bound: float, upper_bound: float) -> float:
        return RangePartitioner._ceil_step(
            (upper_bound - lower_bound) / (self.max_partitions - 1)
        )

    @property
    def _partition_center_offset(self) -> float:
        return self._step / 2


class IntegerRangePartitioner(RangePartitioner[int]):
    """
    Partition integer values in adjacent intervals of the same length.

    The range of intervals and interval size is computed based on attributes
    :attr:`.max_partitions`, :attr:`.lower_bound`, and :attr:`.upper_bound`.

    Partition boundaries and interval sized are chosen with interpretability in mind and
    are always an integer and a power of 10, or a multiple of 2 or 5 of a power of 10,
    e.g. 1, 2, 5, 10, 20, 50, and so on.

    The intervals also satisfy the following conditions:

    - :attr:`lower_bound` is within the first interval
    - :attr:`upper_bound` is within the last interval

    For example, with :attr:`.max_partitions` = 5, :attr:`.lower_bound` = 3, and
    :attr:`.upper_bound` = 11, the resulting partitioning would be:
    [2, 4), [4, 6), [6, 8), [8, 10), [10, 12)
    """

    def _step_size(self, lower_bound: int, upper_bound: int) -> int:
        return max(
            1,
            int(
                RangePartitioner._ceil_step(
                    (upper_bound - lower_bound) / (self.max_partitions - 1)
                )
            ),
        )

    @property
    def _partition_center_offset(self) -> int:
        return self._step // 2


@inheritdoc(match="[see superclass]")
class CategoryPartitioner(Partitioner[T_Values]):
    """
    Partition categorical values.

    Create one partition each per unique value, considering only the
    :attr:`.max_partitions` most frequent values.
    """

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._frequencies is not None

    @property
    def is_categorical(self) -> bool:
        """
        ``True``
        """
        return True

    # noinspection PyMissingOrEmptyDocstring
    def fit(self: T_Self, values: Iterable[T_Values], **fit_params: Any) -> T_Self:
        """[see superclass]"""

        self: CategoryPartitioner  # support type hinting in PyCharm

        values = self._as_non_empty_array(values)

        partitions, frequencies = np.unique(values, return_counts=True)
        order_descending = np.flip(np.argsort(frequencies))[: self.max_partitions]

        self._partitions = partitions[order_descending]
        self._frequencies = frequencies[order_descending]

        return self


__tracker.validate()
