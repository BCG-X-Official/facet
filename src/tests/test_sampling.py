import numpy as np

from yieldengine.sampling import (
    ContinuousValuePartitioning,
    DEFAULT_MAX_PARTITIONS,
    DiscreteValuePartitioning,
)


def test_discrete_partitioning() -> None:
    for i in range(0, 10000):
        values = np.random.randint(
            low=0, high=1000, size=max(int(np.random.rand() * 1000), 2)
        )
        dvp = DiscreteValuePartitioning(
            values=values, max_partitions=DEFAULT_MAX_PARTITIONS
        )
        # test correct number of partitions
        assert dvp.n_partitions <= DEFAULT_MAX_PARTITIONS
        assert dvp.n_partitions == len(dvp.partitions())


def test_continouus_partitioning() -> None:
    for i in range(0, 10000):
        values = (
            np.random.randint(
                low=0, high=1000, size=max(int(np.random.rand() * 1000), 2)
            )
            * np.random.rand()
        )
        cvp = ContinuousValuePartitioning(
            values=values, max_partitions=DEFAULT_MAX_PARTITIONS
        )
        # test correct number of partitions
        assert cvp.n_partitions <= DEFAULT_MAX_PARTITIONS
        assert cvp.n_partitions == len(cvp.partitions())
