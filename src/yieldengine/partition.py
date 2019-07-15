import logging
import math
from abc import ABC, ABCMeta, abstractmethod
from typing import *

import numpy as np
import pandas as pd

from yieldengine import ListLike

log = logging.getLogger(__name__)

DEFAULT_MAX_PARTITIONS = 20

ValueType = TypeVar("ValueType")
NumericType = TypeVar("NumericType", bound=Union[int, float])


class Partitioning(ABC, Generic[ValueType]):
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
    Partitioning[NumericType], Generic[NumericType], metaclass=ABCMeta
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
        step = self._step
        return (
            self._first_partition + (idx * step) for idx in range(0, self.n_partitions)
        )

    def frequencies(self) -> Iterable[int]:
        return self._frequencies

    @property
    def n_partitions(self) -> int:
        return self._n_partitions

    def partition_bounds(self) -> Iterable[Tuple[NumericType, NumericType]]:
        center_offset_left = self._partition_center_offset
        center_offset_right = self._step - center_offset_left
        return (
            (center - center_offset_left, center + center_offset_right)
            for center in self.partitions()
        )

    @property
    def partition_width(self) -> NumericType:
        return self._step

    @staticmethod
    def _ceil_step(step: float):
        """
        :param step: the step size to round by
        :return: the nearest step size in the series
                 (..., 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, ...)
        """
        if step <= 0:
            raise ValueError("arg step must be positive")

        return min(10 ** math.ceil(math.log10(step * m)) / m for m in [1, 2, 5])

    @staticmethod
    @abstractmethod
    def _step_size(
        lower_bound: NumericType, upper_bound: NumericType, max_partitions: int
    ) -> NumericType:
        pass

    @property
    @abstractmethod
    def _partition_center_offset(self) -> NumericType:
        pass


class ContinuousRangePartitioning(RangePartitioning[float]):
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
    @staticmethod
    def _step_size(lower_bound: int, upper_bound: int, max_partitions: int) -> int:
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
        return self._partitions

    def frequencies(self) -> Iterable[int]:
        return self._frequencies

    @property
    def n_partitions(self) -> int:
        return len(self._partitions)
