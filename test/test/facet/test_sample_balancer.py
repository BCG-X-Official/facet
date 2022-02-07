""" Test facet.data.SampleBalancer"""
import logging

import numpy as np
import pandas as pd
import pytest

from facet.data import Sample, SampleBalancer

log = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def set_numpy_seed() -> None:
    np.random.seed(42)
    yield


@pytest.fixture(scope="module")
def binary_target() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c1": np.floor(np.random.random(1000) * 1000),
            "c2": np.floor(np.random.random(1000) * 1000),
            "c3": np.floor(np.random.random(1000) * 1000),
            "target": ([0] * 900) + ([1] * 100),
        }
    )


@pytest.fixture(scope="module")
def multiclass_target() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c1": np.floor(np.random.random(1000) * 1000),
            "c2": np.floor(np.random.random(1000) * 1000),
            "c3": np.floor(np.random.random(1000) * 1000),
            "target": (["a"] * 700) + (["b"] * 200) + (["c"] * 100),
        }
    )


def test_init_argument_validation() -> None:
    with pytest.raises(
        expected_exception=ValueError, match="oversample and undersample are both false"
    ):
        SampleBalancer(
            undersample=False, oversample=False, target_frequencies={"a": 0.1}
        )

    with pytest.raises(
        expected_exception=ValueError, match="target_frequencies is empty"
    ):
        SampleBalancer(target_frequencies={})

    for faulty in (-0.1, 1.0, 0.0, 1.0001):
        with pytest.raises(
            expected_exception=ValueError,
            match="target frequency for label a expected in range",
        ):
            SampleBalancer(target_frequencies={"a": faulty})

    with pytest.raises(
        expected_exception=TypeError,
        match="target frequency value for label a should be float but is a str",
    ):
        # noinspection PyTypeChecker
        SampleBalancer(target_frequencies={"a": "a"})


def test_frequency_validation(multiclass_target: pd.DataFrame) -> None:

    # 1. all frequencies listed, but cumulatively <> 1.00
    test_sample = Sample(observations=multiclass_target, target_name="target")
    with pytest.raises(
        expected_exception=ValueError,
        match="target_frequencies expects a cumulative frequency of 1.0, but is 0.99",
    ):
        sb = SampleBalancer(target_frequencies={"a": 0.5, "b": 0.25, "c": 0.24})
        sb.balance(sample=test_sample)

    # this should be accepted, since we round to two digits for the check
    sb = SampleBalancer(target_frequencies={"a": 0.333, "b": 0.333, "c": 0.333})
    sb.balance(sample=test_sample)

    # 2. a frequency is listed with a class label, which does not exist in the data
    # 2.1 one extra unknown class label
    with pytest.raises(
        expected_exception=ValueError,
        match="target_frequencies specifies unknown class labels: d",
    ):
        sb = SampleBalancer(target_frequencies={"a": 0.5, "b": 0.25, "d": 0.25})
        sb.balance(sample=test_sample)
    # 2.2 several extra unknown class labels
    with pytest.raises(
        expected_exception=ValueError,
        match="target_frequencies specifies unknown class labels: d,e",
    ):
        sb = SampleBalancer(
            target_frequencies={"a": 0.5, "b": 0.25, "d": 0.20, "e": 0.05}
        )
        sb.balance(sample=test_sample)

    # 3. the sample has an extra class label for which the target frequency has not
    #    been defined in the balancer, but cumulative defined target frequency is
    #    already 1.00
    with pytest.raises(
        expected_exception=ValueError,
        match=r"cumulative frequency of 1.0, but class label\(s\) c have been omitted",
    ):
        sb = SampleBalancer(
            target_frequencies={
                "a": 0.5,
                "b": 0.5,
            }
        )
        sb.balance(sample=test_sample)


def test_undersample_with_binary_labels(binary_target) -> None:
    test_sample = Sample(observations=binary_target, target_name="target")

    test_balancer = SampleBalancer(
        target_frequencies={0: 0.6, 1: 0.4}, undersample=True, oversample=False
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    print(balanced.target.value_counts())
    assert value_counts[0] / value_counts[1] == pytest.approx(0.6 / 0.4, abs=0.02)

    log.info(value_counts)


def test_oversample_with_binary_labels(binary_target) -> None:

    test_sample = Sample(observations=binary_target, target_name="target")

    test_balancer = SampleBalancer(
        target_frequencies={0: 0.6, 1: 0.4}, undersample=False, oversample=True
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    print(balanced.target.value_counts())
    assert value_counts[0] / value_counts[1] == pytest.approx(0.6 / 0.4, abs=0.02)

    log.info(value_counts)


def test_mixed_sample_with_binary_labels(binary_target) -> None:

    test_sample = Sample(observations=binary_target, target_name="target")

    test_balancer = SampleBalancer(
        target_frequencies={0: 0.6, 1: 0.4}, undersample=True, oversample=True
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()
    print(balanced.target.value_counts())
    assert value_counts[0] / value_counts[1] == pytest.approx(0.6 / 0.4, abs=0.02)

    log.info(value_counts)


def test_undersample_with_multilabel(multiclass_target) -> None:
    test_sample = Sample(observations=multiclass_target, target_name="target")

    test_balancer = SampleBalancer(
        target_frequencies={"a": 0.4, "b": 0.3, "c": 0.3},
        undersample=True,
        oversample=False,
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()

    assert value_counts[1] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)

    assert value_counts[2] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)
    log.info(value_counts)


def test_oversample_with_multilabel(multiclass_target) -> None:

    test_sample = Sample(observations=multiclass_target, target_name="target")

    test_balancer = SampleBalancer(
        target_frequencies={"a": 0.4, "b": 0.3, "c": 0.3},
        undersample=False,
        oversample=True,
    )

    balanced = test_balancer.balance(test_sample)
    value_counts = balanced.target.value_counts()

    assert value_counts[1] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)

    assert value_counts[2] / value_counts[0] == pytest.approx(0.3 / 0.4, abs=0.05)
    log.info(value_counts)


def test_no_change(binary_target) -> None:
    test_sample = Sample(observations=binary_target, target_name="target")

    test_balancer = SampleBalancer(
        target_frequencies={0: 0.9},
    )

    balanced = test_balancer.balance(test_sample)

    assert test_sample == balanced
    value_counts = balanced.target.value_counts()
    assert value_counts[1] / value_counts[0] == pytest.approx(0.11, abs=0.01)

    log.info(value_counts)
