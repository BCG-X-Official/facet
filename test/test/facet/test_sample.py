from typing import Any, List, cast

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from pandas.testing import assert_frame_equal

from facet.data import Sample


def test_sample_init(california_df: pd.DataFrame, california_target: str) -> None:
    # check handling of various invalid inputs

    # 1. sample parameter
    # 1.1 None
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=None, target_name=california_target)

    # 1.2 not a DF
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=[], target_name=california_target)

    # 2. no valid target specified
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        Sample(observations=california_df, target_name=None)  # type: ignore

    # store list of feature columns:
    f_columns = list(california_df.columns)
    f_columns.remove(california_target)

    # 2.1 invalid feature column specified
    with pytest.raises(KeyError):
        f_columns_invalid = f_columns.copy()
        f_columns_invalid.append("doesnt_exist")
        Sample(
            observations=california_df,
            feature_names=f_columns_invalid,
            target_name=california_target,
        )

    # 2.2 invalid target column specified
    with pytest.raises(KeyError):
        Sample(
            observations=california_df,
            feature_names=f_columns,
            target_name="doesnt_exist",
        )

    # 3. column is target and also feature
    with pytest.raises(KeyError):
        f_columns_invalid = f_columns.copy()
        f_columns_invalid.append(california_target)
        Sample(
            observations=california_df,
            feature_names=f_columns_invalid,
            target_name=california_target,
        )

    # 4. weight column is not defined
    with pytest.raises(KeyError):
        Sample(
            observations=california_df,
            target_name=california_target,
            weight_name="doesnt_exist",
        )


def test_sample(california_df: pd.DataFrame, california_target: str) -> None:
    # define various assertions we want to test:
    def run_assertions(sample: Sample) -> None:
        assert sample.target.name == california_target
        assert sample.weight is not None
        assert sample.weight.name == california_target
        assert california_target not in sample.feature_names
        assert len(sample.feature_names) == len(california_df.columns) - 1

        assert type(sample.target) == pd.Series
        assert type(sample.weight) == pd.Series
        assert type(sample.features) == pd.DataFrame

        assert len(sample.target) == len(sample.features)

    # test explicit setting of all fields
    feature_columns = list(california_df.drop(columns=california_target).columns)
    s = Sample(
        observations=california_df,
        target_name=california_target,
        feature_names=feature_columns,
        weight_name=california_target,
    )

    # _rank_learners the checks on s:
    run_assertions(s)

    # test implicit setting of features by only giving the target
    s2 = Sample(
        observations=california_df,
        target_name=california_target,
        weight_name=california_target,
    )

    # _rank_learners the checks on s2:
    run_assertions(s2)

    # test numerical features
    features_numerical = s.features.select_dtypes(np.number).columns
    assert "HouseAge" in features_numerical

    # test categorical features
    features_non_numerical = s.features.select_dtypes(object).columns
    assert len(features_non_numerical) == 0

    # assert feature completeness
    assert (
        len(
            set(features_numerical)
            .union(set(features_non_numerical))
            .difference(s.feature_names)
        )
        == 0
    )

    # test length
    assert len(s) == len(california_df)

    # test select_observations
    sub = s2.subsample(iloc=[0, 1, 2, 3])
    assert len(sub) == 4

    # test subset of features
    assert_frame_equal(
        s2.keep(feature_names=s2.feature_names[:10]).features, s2.features.iloc[:, :10]
    )

    with pytest.raises(ValueError):
        s2.keep(feature_names=["does not exist"])

    # test that s.features is a deterministic operation that does not depend on the
    # global python environment variable PYTHONHASHSEED
    parallel = Parallel(n_jobs=-3)

    def get_column(sample: Sample) -> List[Any]:
        return cast(List[Any], sample.features.columns.to_list())

    columns1, columns2 = parallel(delayed(get_column)(sample) for sample in [s, s])
    assert columns1 == columns2

    # creating a sample with non-string column names raises an exception
    with pytest.raises(
        TypeError,
        match=(
            "^all column names in arg observations must be strings, "
            "but included: int$"
        ),
    ):
        Sample(
            california_df.set_axis([*california_df.columns[1:], 1], axis=1),
            target_name=california_target,
        )
