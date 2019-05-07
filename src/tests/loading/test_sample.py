import pandas as pd
import pytest

# noinspection PyUnresolvedReferences
from tests.shared_fixtures import batch_table as test_sample_data
from yieldengine.loading.sample import Sample


# checks various erroneous inputs
def test_sample_init(test_sample_data):
    # 1. sample parameter
    # 1.1 None
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=None, target_name="target")

    # 1.2 not a DF
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=[], target_name="target")

    # 2. no features and no target specified
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Sample(observations=test_sample_data, target_name=None)

    # store list of feature columns:
    f_columns = list(test_sample_data.columns)
    f_columns.remove("Yield")

    # 2.1 invalid feature column specified
    with pytest.raises(ValueError):
        f_columns_false = f_columns.copy()
        f_columns_false.append("doesnt_exist")
        Sample(
            observations=test_sample_data,
            feature_names=f_columns_false,
            target_name="Yield",
        )

    # 2.2 invalid target column specified
    with pytest.raises(ValueError):
        Sample(
            observations=test_sample_data,
            target_name="doesnt_exist",
            feature_names=f_columns,
        )

    # 3. column is target and also feature
    with pytest.raises(ValueError):
        f_columns_false = f_columns.copy()
        f_columns_false.append("Yield")

        Sample(
            observations=test_sample_data,
            feature_names=f_columns_false,
            target_name="Yield",
        )


def test_sample(test_sample_data):
    # define various assertions we want to test:
    def run_assertions(s: Sample):
        assert s.target_name == "Yield"
        assert "Yield" not in s.feature_names
        assert len(s.feature_names) == len(test_sample_data.columns) - 1

        assert type(s.target) == pd.Series
        assert type(s.features) == pd.DataFrame

        assert len(s.target) == len(s.features)

    # test explicit setting of both target & features
    feature_columns = list(test_sample_data.drop(columns="Yield").columns)
    s = Sample(
        observations=test_sample_data,
        target_name="Yield",
        feature_names=feature_columns,
    )

    # run the checks on s:
    run_assertions(s)

    # test implicit setting of features by only giving the target
    s2 = Sample(observations=test_sample_data, target_name="Yield")

    # run the checks on s2:
    run_assertions(s2)

    # test numerical features
    assert (
        "Step4 Fermentation Sensor Data Phase2 Pressure Val04 (mbar)"
        in s.features_numerical
    )

    # test categorical features
    assert "Step4 RawMat Internal Compound01 QC (id)" in s.features_categorical

    # assert feature completeness
    assert (
        len(
            set(s.features_numerical)
            .union(set(s.features_categorical))
            .difference(s.feature_names)
        )
        == 0
    )

    # test length
    assert len(s) == len(test_sample_data)
