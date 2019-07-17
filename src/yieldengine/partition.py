"""Partition sets.

A :class:`~Partitioning` partitions a set of values
into finitely many partitions (synonym of buckets).
The method :meth:`~Partitioning.frequencies` returns  an iterable of the number of
values in the different partitions, the method :meth:`~Partitioning.partitions`
returns a list of central value in each partition,
and :attr:`~Partitioning.n_partitions` is the number of partitions.


"""
import logging
from abc import ABC, ABCMeta, abstractmethod
from typing import *

import math
import numpy as np
import pandas as pd

from yieldengine import ListLike

log = logging.getLogger(__name__)

DEFAULT_MAX_PARTITIONS = 20

ValueType = TypeVar("ValueType")
NumericType = TypeVar("NumericType", bound=Union[int, float])


class Partitioning(ABC, Generic[ValueType]):
    """Partition a set of values, for use in visualizations and simulations."""

    @abstractmethod
    def partitions(self) -> Iterable[ValueType]:
        """Return central values of the partitions.

        :return: for each partition, return a central value representing the partition
        """
        pass

    @abstractmethod
    def frequencies(self) -> Iterable[int]:
        """Return the number of elements in each partitions.

        :return: for each partition, the number of observed values that fall within
          the partition
        """

    @property
    @abstractmethod
    def n_partitions(self) -> int:
        """Number of partitions."""
        pass


class RangePartitioning(
    Partitioning[NumericType], Generic[NumericType], metaclass=ABCMeta
):
    """
    Partition numerical values in successive intervals of the same length.

    The partitions are made of intervals which have all the same lengths.

    The interval length is computed based on
    :attr:`max_partitions`, :attr:`lower_bound` and :attr:`upper_bound` by
    :meth:`_step_size`.


    Each partition is an interval whose endpoints are multiple of the interval
    length. The rules used to determine these intervals are that:
    - :attr:`lower_bound` is in the first interval
    - :attr:`upper_bound` is in the last interval

    For example, if the computed interval length is 0.2, some possible
    partitions would be:
    [3.2, 3.4), [3.4, 3.6), [3.6, 3.8), [4.0, 4.2), [4.4, 4.6), [4.6, 4.8]

    Implementations must define :meth:`_step_size` and :meth:`_partition_center_offset`.



    :param values: list like of values to partition
    :param int max_partitions: the max number of partitions to make (default = 20);
      it should be greater or equal than 2
    :param lower_bound: the lower bound of the elements in the partition
    :param upper_bound: the upper bound of the elements in the partition
    """

    def __init__(
        self,
        values: ListLike[NumericType],
        max_partitions: int = DEFAULT_MAX_PARTITIONS,
        lower_bound: Optional[NumericType] = None,
        upper_bound: Optional[NumericType] = None,
    ) -> None:
        """Constructor

        :param values: list like of values to partition
        :param int max_partitions: the max number of partitions to make (default = 20);
          it should be greater or equal than 2
        :param lower_bound: the lower bound of the elements in the partition
        :param upper_bound: the upper bound of the elements in the partition

        """

        super().__init__()

        if lower_bound is None:
            lower_bound = np.min(values)
        if upper_bound is None:
            upper_bound = np.max(values)
        if lower_bound >= upper_bound:
            raise ValueError(
                f"arg lower_bound >= arg upper_bound: [{lower_bound}, {upper_bound})"
            )

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
            """Return the number of elements in each partitions."""
            partition_indices = [
                int(round(value - first_partition) / step) for value in values
            ]
            frequencies = [0] * n_partitions
            for idx in partition_indices:
                if 0 <= idx < n_partitions:
                    frequencies[idx] += 1

            return frequencies

        self._frequencies = _frequencies()

    def partitions(self) -> Iterable[NumericType]:
        """Return the central values of the partitions.

        :return: for each partition, return a central value representing the partition
        """
        step = self._step
        return (
            self._first_partition + (idx * step) for idx in range(0, self.n_partitions)
        )

    def frequencies(self) -> Iterable[int]:
        """Return the number of elements in each partitions.

        :return: for each partition, the number of observed values that fall within
          the partition
        """
        return self._frequencies

    @property
    def n_partitions(self) -> int:
        """Number of partitions."""
        return self._n_partitions

    def partition_bounds(self) -> Iterable[Tuple[NumericType, NumericType]]:
        """Return the endpoints of the intervals making the partitions.

        :return: generator of the tuples (x, y) where x and y and the endopoints of
          the partitions
        """

        center_offset_left = self._partition_center_offset
        center_offset_right = self._step - center_offset_left
        return (
            (center - center_offset_left, center + center_offset_right)
            for center in self.partitions()
        )

    @property
    def partition_width(self) -> NumericType:
        """The interval length."""
        return self._step

    @staticmethod
    def _ceil_step(step: float):
        """
        Round the step size (arbitrary float) to a human-readable number like 0.5, 1, 2.

        :param step: the step size to round by
        :return: the nearest greater or equal step size in the series
                 (..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, ...)
        >>> from yieldengine.partition import RangePartitioning
        >>> RangePartitioning._ceil_step(1.99)
        2.0
        >>> RangePartitioning._ceil_step(2.)
        2.0
        >>> RangePartitioning._ceil_step(2.01)
        5.0
        >>> RangePartitioning._ceil_step(4.9)
        5.0
        >>> RangePartitioning._ceil_step(5.1)
        10.0
        """
        if step <= 0:
            raise ValueError("arg step must be positive")

        return min(10 ** math.ceil(math.log10(step * m)) / m for m in [1, 2, 5])

    @staticmethod
    @abstractmethod
    def _step_size(
        lower_bound: NumericType, upper_bound: NumericType, max_partitions: int
    ) -> NumericType:
        """Compute the step size (interval length) used in the partitions."""
        pass

    @property
    @abstractmethod
    def _partition_center_offset(self) -> NumericType:
        """Offset between center and endpoints of an interval."""
        pass


class ContinuousRangePartitioning(RangePartitioning[float]):
    """
    Partition numerical values in successive intervals of the same length.

    The partitions are made of intervals which have all the same length which is a
    number in the series
    (..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, ...)

    The interval length is computed based on
    :attr:`max_partitions`, :attr:`lower_bound` and :attr:`upper_bound` by
    :meth:`_step_size`.


    Each partition is an interval whose endpoints are multiple of the interval
    length. The rules used to determine these intervals are that:
    - :attr:`lower_bound` is in the first interval
    - :attr:`upper_bound` is in the last interval

    For example, if the computed interval length is 0.2, some possible
    partitions would be:
    [3.2, 3.4), [3.4, 3.6), [3.6, 3.8), [4.0, 4.2), [4.4, 4.6), [4.6, 4.8]

    Implementations must define :meth:`_step_size` and :meth:`_partition_center_offset`.

    """

    @staticmethod
    def _step_size(
        lower_bound: float, upper_bound: float, max_partitions: int
    ) -> float:
        return RangePartitioning._ceil_step(
            (upper_bound - lower_bound) / (max_partitions - 1)
        )

    @property
    def _partition_center_offset(self) -> float:
        return self._step / 2


class IntegerRangePartitioning(RangePartitioning[int]):
    """
    Partition numerical values in successive intervals of same length with int bounds.

    The interval length of the intervals making the partitions is an integer. The
    bounds of the intervals are all integer which are multiple of the interval length.

    The interval length is computed based on
    :attr:`max_partitions`, :attr:`lower_bound` and :attr:`upper_bound` by
    :meth:`_step_size`.


    Each partition is an interval whose endpoints are multiple of the interval
    length. The rules used to determine these intervals are that:
    - :attr:`lower_bound` is in the first interval
    - :attr:`upper_bound` is in the last interval

    For example, if the computed interval length is 0.2, some possible
    partitions would be:
    [3.2, 3.4), [3.4, 3.6), [3.6, 3.8), [4.0, 4.2), [4.4, 4.6), [4.6, 4.8]

    Implementations must define :meth:`_step_size` and :meth:`_partition_center_offset`.



    :param values: list like of values to partition
    :param int max_partitions: the max number of partitions to make (default = 20);
      it should be greater or equal than 2
    :param lower_bound: the lower bound of the elements in the partition
    :param upper_bound: the upper bound of the elements in the partition
    """

    @staticmethod
    def _step_size(lower_bound: int, upper_bound: int, max_partitions: int) -> int:
        """Compute the step size of the central values."""
        return max(
            1,
            int(
                RangePartitioning._ceil_step(
                    (upper_bound - lower_bound) / (max_partitions - 1)
                )
            ),
        )

    @property
    def _partition_center_offset(self) -> int:
        return self._step // 2


class CategoryPartitioning(Partitioning[ValueType]):
    """Partition categorical values.

    Partition the elements by their values, keeping only the :attr:`max_partitions`
    msot frequent values.

    :param values: list of values
    :max_partitions: the maximum number of partitions
    """

    def __init__(
        self, values: ListLike[ValueType], max_partitions: int = DEFAULT_MAX_PARTITIONS
    ) -> None:
        super().__init__()

        value_counts = pd.Series(data=values).value_counts(ascending=False)
        if len(value_counts) > max_partitions:
            log.warning(
                f"arg values has {len(value_counts)} unique values, but "
                f"arg max_partitions is only {max_partitions}"
            )
        self._frequencies = value_counts.values[:max_partitions]
        self._partitions = value_counts.index.values[:max_partitions]

    def partitions(self) -> Iterable[ValueType]:
        """The list of the :attr:`max_partitions` most frequent values.

        :return: the list of most frequent values, ordered decreasingly by the
          frequency"""
        return self._partitions

    def frequencies(self) -> Iterable[int]:
        """
        Return the number of elements in each partitions.

        :return: for each partition, the number of observed values that fall within
          the partition
        """
        return self._frequencies

    @property
    def n_partitions(self) -> int:
        """Number of partitions."""
        return len(self._partitions)
