from typing import *

import numpy as np

from gamma.yieldengine.partition import (
    ContinuousRangePartitioner,
    DEFAULT_MAX_PARTITIONS,
    IntegerRangePartitioner,
)


def test_discrete_partitioning() -> None:
    for i in range(0, 10000):
        values: Sequence[int] = np.random.randint(
            low=0, high=10000, size=max(int(np.random.rand() * 1000), 3)
        )
        # noinspection PyTypeChecker
        dvp = IntegerRangePartitioner(max_partitions=DEFAULT_MAX_PARTITIONS).fit(
            values=values
        )
        # test correct number of partitions
        assert len(dvp) <= DEFAULT_MAX_PARTITIONS
        partitions = list(dvp.partitions())
        assert len(dvp) == len(partitions)


def test_continuous_partitioning() -> None:
    for i in range(0, 10000):
        values = (
            np.random.randint(
                low=0, high=10000, size=max(int(np.random.rand() * 1000), 3)
            )
            * np.random.rand()
        )
        cvp = ContinuousRangePartitioner(max_partitions=DEFAULT_MAX_PARTITIONS).fit(
            values=values
        )
        # test correct number of partitions
        assert len(cvp) <= DEFAULT_MAX_PARTITIONS
        partitions = list(cvp.partitions())
        assert len(cvp) == len(partitions)
