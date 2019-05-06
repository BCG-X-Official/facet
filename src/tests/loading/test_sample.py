from tests.shared_fixtures import batch_table as test_sample_data
from yieldengine.loading.sample import Sample
import pytest
import pandas as pd

# checks various erroneous inputs
def test_sample_init(test_sample_data):
    # 1. sample parameter
    # 1.1 None
    with pytest.raises(ValueError):
        Sample(sample=None)

    # 1.2 not a DF
    with pytest.raises(ValueError):
        Sample(sample=[])

    # 2. no features and no target specified
    with pytest.raises(ValueError):
        Sample(sample=test_sample_data)

    # store list of feature columns:
    f_columns = list(test_sample_data.columns)
    f_columns.remove("Yield")

    # 2.1 invalid feature column specified
    with pytest.raises(ValueError):
        f_columns_false = f_columns.copy()
        f_columns_false.append("doesnt_exist")
        Sample(sample=test_sample_data, features=f_columns_false, target="Yield")

    # 2.2 invalid target column specified
    with pytest.raises(ValueError):
        Sample(sample=test_sample_data, target="doesnt_exist", features=f_columns)

    # 3. column is target and also feature
    with pytest.raises(ValueError):
        f_columns_false = f_columns.copy()
        f_columns_false.append("Yield")

        Sample(sample=test_sample_data, features=f_columns_false, target="Yield")


def test_sample(test_sample_data):
    # define various assertions we want to test:
    def run_assertions(s: Sample):
        assert s.target == "Yield"
        assert "Yield" not in s.features
        assert len(s.features) == len(test_sample_data.columns) - 1

        assert type(s.target_data) == pd.Series
        assert type(s.feature_data) == pd.DataFrame

        assert len(s.target_data) == len(s.feature_data)

    # test explicit setting of both target & features
    feature_columns = list(test_sample_data.drop(columns="Yield").columns)
    s = Sample(sample=test_sample_data, target="Yield", features=feature_columns)

    # run the checks on s:
    run_assertions(s)

    # test implicit setting of features by only giving the target
    s2 = Sample(sample=test_sample_data, target="Yield")

    # run the checks on s2:
    run_assertions(s2)

    # property of s.features should not be mutable
    with pytest.raises(AttributeError):
        s.features = "error"

    # property of s.target should not be mutable
    with pytest.raises(AttributeError):
        s.target = "error"

    # test numerical features
    assert (
        "Step4 Fermentation Sensor Data Phase2 Pressure Val04 (mbar)"
        in s.numerical_features
    )

    # test categorical features
    assert "Step4 RawMat Internal Compound01 QC (id)" in s.categorical_features

    # assert feature completeness
    assert (
        len(
            set(s.numerical_features)
            .union(set(s.categorical_features))
            .difference(s.features)
        )
        == 0
    )

    # test length
    assert len(s) == len(test_sample_data)
