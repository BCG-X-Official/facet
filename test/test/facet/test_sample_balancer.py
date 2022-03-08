""" Test facet.data.SampleBalancer"""
import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from facet.data import (
    Sample,
    SampleBalancer,
    TargetFrequencySampleBalancer,
    UniformSampleBalancer,
)

log = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def set_numpy_seed() -> None:
    np.random.seed(42)
    yield


@pytest.fixture(scope="module")
def binary_target() -> Sample:
    return Sample(
        observations=pd.DataFrame(
            data={
                "c1": np.floor(np.random.random(1000) * 1000),
                "c2": np.floor(np.random.random(1000) * 1000),
                "c3": np.floor(np.random.random(1000) * 1000),
                "target": ([0] * 900) + ([1] * 100),
            }
        ),
        target_name="target",
    )


@pytest.fixture(scope="module")
def multiclass_target() -> Sample:
    return Sample(
        observations=pd.DataFrame(
            data={
                "c1": np.floor(np.random.random(1000) * 1000),
                "c2": np.floor(np.random.random(1000) * 1000),
                "c3": np.floor(np.random.random(1000) * 1000),
                "target": (["a"] * 700) + (["b"] * 200) + (["c"] * 100),
            }
        ),
        target_name="target",
    )


def test_init_argument_validation() -> None:
    with pytest.raises(
        expected_exception=ValueError, match="oversample and undersample are both false"
    ):
        TargetFrequencySampleBalancer(
            undersample=False, oversample=False, target_frequencies={"a": 0.1}
        )

    with pytest.raises(
        expected_exception=ValueError, match="target_frequencies is empty"
    ):
        TargetFrequencySampleBalancer(target_frequencies={})

    for faulty in (-0.1, 1.0, 0.0, 1.0001):
        with pytest.raises(
            expected_exception=ValueError,
            match="target frequency for label a expected in range",
        ):
            TargetFrequencySampleBalancer(target_frequencies={"a": faulty})

    with pytest.raises(
        expected_exception=TypeError,
        match="target frequency value for label a should be float but is a str",
    ):
        # noinspection PyTypeChecker
        TargetFrequencySampleBalancer(target_frequencies={"a": "a"})


def test_frequency_validation(multiclass_target: Sample) -> None:

    # 1. all frequencies listed, but cumulatively <> 1.00
    with pytest.raises(
        expected_exception=ValueError,
        match="target_frequencies expects a cumulative frequency of 1.0, but is 0.99",
    ):
        sb = TargetFrequencySampleBalancer(
            target_frequencies={"a": 0.5, "b": 0.25, "c": 0.24}
        )
        sb.balance(sample=multiclass_target)

    # this should be accepted, since we round to two digits for the check
    sb = TargetFrequencySampleBalancer(
        target_frequencies={"a": 0.333, "b": 0.333, "c": 0.333}
    )
    sb.balance(sample=multiclass_target)

    # 2. a frequency is listed with a class label, which does not exist in the data
    # 2.1 one extra unknown class label
    with pytest.raises(
        expected_exception=ValueError,
        match="target_frequencies specifies unknown class labels: d",
    ):
        sb = TargetFrequencySampleBalancer(
            target_frequencies={"a": 0.5, "b": 0.25, "d": 0.25}
        )
        sb.balance(sample=multiclass_target)
    # 2.2 several extra unknown class labels
    with pytest.raises(
        expected_exception=ValueError,
        match="target_frequencies specifies unknown class labels: d,e",
    ):
        sb = TargetFrequencySampleBalancer(
            target_frequencies={"a": 0.5, "b": 0.25, "d": 0.20, "e": 0.05}
        )
        sb.balance(sample=multiclass_target)

    # 3. the sample has an extra class label for which the target frequency has not
    #    been defined in the balancer, but cumulative defined target frequency is
    #    already 1.00
    with pytest.raises(
        expected_exception=ValueError,
        match=r"cumulative frequency of 1.0, but class label\(s\) c have been omitted",
    ):
        sb = TargetFrequencySampleBalancer(
            target_frequencies={
                "a": 0.5,
                "b": 0.5,
            }
        )
        sb.balance(sample=multiclass_target)


def test_undersample_with_binary_labels(binary_target: Sample) -> None:

    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={0: 0.6, 1: 0.4}, undersample=True, oversample=False
    )

    balanced = test_balancer.balance(binary_target)
    value_counts = balanced.target.value_counts()
    print(balanced.target.value_counts())
    assert value_counts[0] / value_counts[1] == pytest.approx(0.6 / 0.4, abs=0.02)

    log.info(value_counts)


def test_oversample_with_binary_labels(binary_target: Sample) -> None:
    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={0: 0.6, 1: 0.4}, undersample=False, oversample=True
    )

    balanced = test_balancer.balance(binary_target)
    value_counts = balanced.target.value_counts()
    print(balanced.target.value_counts())
    assert value_counts[0] / value_counts[1] == pytest.approx(0.6 / 0.4, abs=0.02)

    log.info(value_counts)


def test_mixed_sample_with_binary_labels(binary_target: Sample) -> None:

    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={0: 0.6, 1: 0.4}, undersample=True, oversample=True
    )

    balanced = test_balancer.balance(binary_target)
    value_counts = balanced.target.value_counts()
    print(balanced.target.value_counts())
    assert value_counts[0] / value_counts[1] == pytest.approx(0.6 / 0.4, abs=0.02)

    log.info(value_counts)


def test_undersample_with_multilabel(multiclass_target: Sample) -> None:

    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={"a": 0.4, "b": 0.3, "c": 0.3},
        undersample=True,
        oversample=False,
    )

    balanced = test_balancer.balance(multiclass_target)
    value_counts = balanced.target.value_counts()

    assert value_counts[1] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)

    assert value_counts[2] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)
    log.info(value_counts)


def test_undersample_error() -> None:
    test_sample = Sample(
        observations=pd.DataFrame(
            data={"c1": [1, 2, 3], "c2": [1, 2, 3], "target": ["a", "b", "c"]}
        ),
        target_name="target",
    )

    # test case 1: here a and b retain both 1 observation, but c drops to 0:
    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={"a": 0.45, "b": 0.45, "c": 0.1},
        undersample=True,
        oversample=True,
    )
    with pytest.raises(
        expected_exception=ValueError,
        match="Undersampling of c to meet target frequencies leads to 0 retained",
    ):
        test_balancer.balance(test_sample)

    # test case 2: both classes b and c drop to 0 remaining observations:
    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={"a": 0.8, "b": 0.1, "c": 0.1},
        undersample=True,
        oversample=False,
    )
    with pytest.raises(
        expected_exception=ValueError,
        match="Undersampling of b,c to meet target frequencies leads to 0 retained",
    ):
        test_balancer.balance(test_sample)


def test_oversample_with_multilabel(multiclass_target: Sample) -> None:

    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={"a": 0.4, "b": 0.3, "c": 0.3},
        undersample=False,
        oversample=True,
    )

    balanced = test_balancer.balance(multiclass_target)
    value_counts = balanced.target.value_counts()

    assert value_counts[1] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)

    assert value_counts[2] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)
    log.info(value_counts)


def test_oversample_with_multilabel_uniform(multiclass_target: Sample) -> None:

    test_balancer = UniformSampleBalancer(
        balance_pct=1.0,
        undersample=False,
        oversample=True,
    )

    balanced = test_balancer.balance(multiclass_target)
    value_counts = balanced.target.value_counts()
    value_freqs = value_counts / value_counts.sum()

    assert value_freqs.min() == pytest.approx(0.33, abs=0.01)
    assert value_freqs.max() == pytest.approx(0.33, abs=0.01)

    log.info(value_counts)


def test_no_change(binary_target: Sample) -> None:
    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={0: 0.9},
    )

    balanced = test_balancer.balance(binary_target)

    assert binary_target == balanced
    value_counts = balanced.target.value_counts()
    assert value_counts[1] / value_counts[0] == pytest.approx(0.11, abs=0.01)

    log.info(value_counts)


def test_balancer_random_state(binary_target: Sample) -> None:
    def _balanced_target(balancer: SampleBalancer) -> pd.Series:
        return balancer.balance(binary_target).target

    test_balancer = TargetFrequencySampleBalancer(
        target_frequencies={0: 0.5}, random_state=42
    )
    assert_series_equal(
        _balanced_target(test_balancer), _balanced_target(test_balancer)
    )

    test_balancer = TargetFrequencySampleBalancer(target_frequencies={0: 0.5})

    assert not _balanced_target(test_balancer).equals(_balanced_target(test_balancer))
