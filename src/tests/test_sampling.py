import numpy as np

from yieldengine.partition import (
    ContinuousRangePartitioning,
    DEFAULT_MAX_PARTITIONS,
    IntegerRangePartitioning,
)


def test_discrete_partitioning() -> None:
    for i in range(0, 10000):
        values = np.random.randint(
            low=0, high=1000, size=max(int(np.random.rand() * 1000), 2)
        )
        dvp = IntegerRangePartitioning(
            values=values, max_partitions=DEFAULT_MAX_PARTITIONS
        )
        # test correct number of partitions
        assert dvp.n_partitions <= DEFAULT_MAX_PARTITIONS
        partitions = list(dvp.partitions())
        assert dvp.n_partitions == len(partitions)


def test_continuous_partitioning() -> None:
    for i in range(0, 10000):
        values = (
            np.random.randint(
                low=0, high=1000, size=max(int(np.random.rand() * 1000), 2)
            )
            * np.random.rand()
        )
        cvp = ContinuousRangePartitioning(
            values=values, max_partitions=DEFAULT_MAX_PARTITIONS
        )
        # test correct number of partitions
        assert cvp.n_partitions <= DEFAULT_MAX_PARTITIONS
        partitions = list(cvp.partitions())
        assert cvp.n_partitions == len(partitions)
