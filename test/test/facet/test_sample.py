import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from pandas.testing import assert_frame_equal

from facet.data import Sample


def test_sample_init(boston_df: pd.DataFrame, boston_target: str) -> None:
    # check handling of various invalid inputs

    # 1. sample parameter
    # 1.1 None
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=None, target_name=boston_target)

    # 1.2 not a DF
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=[], target_name=boston_target)

    # 2. no features and no target specified
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        Sample(observations=boston_df, target_name=None)

    # store list of feature columns:
    f_columns = list(boston_df.columns)
    f_columns.remove(boston_target)

    # 2.1 invalid feature column specified
    with pytest.raises(KeyError):
        f_columns_invalid = f_columns.copy()
        f_columns_invalid.append("doesnt_exist")
        Sample(
            observations=boston_df,
            feature_names=f_columns_invalid,
            target_name=boston_target,
        )

    # 2.2 invalid target column specified
    with pytest.raises(KeyError):
        Sample(
            observations=boston_df,
            feature_names=f_columns,
            target_name="doesnt_exist",
        )

    # 3. column is target and also feature
    with pytest.raises(KeyError):
        f_columns_invalid = f_columns.copy()
        f_columns_invalid.append(boston_target)
        Sample(
            observations=boston_df,
            feature_names=f_columns_invalid,
            target_name=boston_target,
        )

    # 4. weight column is not defined
    with pytest.raises(KeyError):
        Sample(
            observations=boston_df,
            target_name=boston_target,
            weight_name="doesnt_exist",
        )


def test_sample(boston_df: pd.DataFrame, boston_target: str) -> None:
    # define various assertions we want to test:
    def run_assertions(sample: Sample):
        assert sample.target.name == boston_target
        assert sample.weight.name == boston_target
        assert boston_target not in sample.feature_names
        assert len(sample.feature_names) == len(boston_df.columns) - 1

        assert type(sample.target) == pd.Series
        assert type(sample.weight) == pd.Series
        assert type(sample.features) == pd.DataFrame

        assert len(sample.target) == len(sample.features)

    # test explicit setting of all fields
    feature_columns = list(boston_df.drop(columns=boston_target).columns)
    s = Sample(
        observations=boston_df,
        target_name=boston_target,
        feature_names=feature_columns,
        weight_name=boston_target,
    )

    # _rank_learners the checks on s:
    run_assertions(s)

    # test implicit setting of features by only giving the target
    s2 = Sample(
        observations=boston_df, target_name=boston_target, weight_name=boston_target
    )

    # _rank_learners the checks on s2:
    run_assertions(s2)

    # test numerical features
    features_numerical = s.features.select_dtypes(np.number).columns
    assert "LSTAT" in features_numerical

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
    assert len(s) == len(boston_df)

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

    def get_column(sample: Sample):
        return list(sample.features.columns)

    columns1, columns2 = parallel(delayed(get_column)(sample) for sample in [s, s])
    assert columns1 == columns2
