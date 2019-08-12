import numpy as np

from gamma.yieldengine.partition import (
    ContinuousRangePartitioning,
    DEFAULT_MAX_PARTITIONS,
    IntegerRangePartitioning,
)


def test_discrete_partitioning() -> None:
    for i in range(0, 10000):
        values = np.random.randint(
            low=0, high=10000, size=max(int(np.random.rand() * 1000), 3)
        )
        dvp = IntegerRangePartitioning(
            values=values, max_partitions=DEFAULT_MAX_PARTITIONS
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
        cvp = ContinuousRangePartitioning(
            values=values, max_partitions=DEFAULT_MAX_PARTITIONS
        )
        # test correct number of partitions
        assert len(cvp) <= DEFAULT_MAX_PARTITIONS
        partitions = list(cvp.partitions())
        assert len(cvp) == len(partitions)
