import numpy as np
import pytest

from facet.data.partition import (
    CategoryPartitioner,
    ContinuousRangePartitioner,
    IntegerRangePartitioner,
)


def test_discrete_partitioning() -> None:
    np.random.seed(42)

    for i in range(10):

        values = np.random.randint(
            low=0, high=10000, size=np.random.randint(low=100, high=200)
        )

        dvp = IntegerRangePartitioner(
            max_partitions=IntegerRangePartitioner.DEFAULT_MAX_PARTITIONS
        ).fit(values=values)

        # test correct number of partitions
        assert len(dvp.partitions_) <= IntegerRangePartitioner.DEFAULT_MAX_PARTITIONS
        assert dvp.partition_bounds_ == [
            (-500, 500),
            (500, 1500),
            (1500, 2500),
            (2500, 3500),
            (3500, 4500),
            (4500, 5500),
            (5500, 6500),
            (6500, 7500),
            (7500, 8500),
            (8500, 9500),
            (9500, 10500),
        ]
        assert (dvp.frequencies_ > 0).all()
        assert (dvp.frequencies_ < 30).all()
        assert dvp.frequencies_.sum() == len(values)


def test_continuous_partitioning() -> None:
    np.random.seed(42)

    for i in range(10):

        values = np.random.normal(
            loc=3.0, scale=8.0, size=np.random.randint(low=2000, high=4000)
        )

        cvp = ContinuousRangePartitioner(
            max_partitions=ContinuousRangePartitioner.DEFAULT_MAX_PARTITIONS
        ).fit(values=values)

        # test correct number of partitions
        assert len(cvp.partitions_) <= ContinuousRangePartitioner.DEFAULT_MAX_PARTITIONS
        partition_bounds_expected = [
            (-22.5, -17.5),
            (-17.5, -12.5),
            (-12.5, -7.5),
            (-7.5, -2.5),
            (-2.5, 2.5),
            (2.5, 7.5),
            (7.5, 12.5),
            (12.5, 17.5),
            (17.5, 22.5),
            (22.5, 27.5),
        ]
        assert (
            cvp.partition_bounds_ == partition_bounds_expected
            or cvp.partition_bounds_ == partition_bounds_expected[1:]
            or cvp.partition_bounds_ == partition_bounds_expected[:-1]
            or cvp.partition_bounds_ == partition_bounds_expected[1:-1]
        )
        assert cvp.frequencies_.sum() >= len(values) * 0.95


def test_category_partitioning() -> None:
    np.random.seed(42)
    for i in range(10):
        values = np.random.randint(
            low=0, high=10, size=np.random.randint(low=100, high=200)
        )
        cp = CategoryPartitioner(max_partitions=4).fit(values=values)
        # test correct number of partitions
        assert len(cp.partitions_) == len(cp.frequencies_)
        assert len(cp.partitions_) <= 4
        assert len(cp.partitions_) == len(np.unique(cp.partitions_))
        assert all(0 <= p <= 10 for p in cp.partitions_)
        for p, f in zip(cp.partitions_, cp.frequencies_):
            assert sum(values == p) == f


def test_partition_with_invalid_values() -> None:

    with pytest.raises(
        ValueError,
        match="arg values is empty",
    ):
        ContinuousRangePartitioner().fit([])

    with pytest.raises(
        ValueError,
        match="insufficient variance in values; cannot infer partitioning bounds",
    ):
        ContinuousRangePartitioner().fit([1])

    with pytest.raises(
        ValueError,
        match="insufficient variance in values; cannot infer partitioning bounds",
    ):
        ContinuousRangePartitioner().fit([1, 1, 1, 10, 1])

    with pytest.raises(
        ValueError,
        match="arg values is empty",
    ):
        IntegerRangePartitioner().fit([])

    with pytest.raises(
        ValueError,
        match="insufficient variance in values; cannot infer partitioning bounds",
    ):
        IntegerRangePartitioner().fit([1])

    with pytest.raises(
        ValueError,
        match="insufficient variance in values; cannot infer partitioning bounds",
    ):
        IntegerRangePartitioner().fit([1, 1, 1, 10, 1])

    with pytest.raises(
        ValueError,
        match="arg values is empty",
    ):
        CategoryPartitioner().fit([])

    CategoryPartitioner().fit([1])

    CategoryPartitioner().fit([1, 1, 1, 10, 1])
