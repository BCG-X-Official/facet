from tests.shared_fixtures import test_sample as test_sample_data
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
    feature_columns = list(test_sample_data.drop(columns="Yield").columns)
    s = Sample(sample=test_sample_data, target="Yield", features=feature_columns)

    assert s.target == "Yield"
    assert "Yield" not in s.features
    assert len(s.features) == len(test_sample_data.columns) - 1

    assert type(s.target_data) == pd.Series
    assert type(s.feature_data) == pd.DataFrame

    assert len(s.target_data) == len(s.feature_data)

    # features property should be settable
    len_before = len(s.features)
    f = s.features[:-2]
    s.features = f
    assert len(s.features) == len(f) == (len_before - 2)

    # features property needs to also check parameters
    with pytest.raises(ValueError):
        f_old = s.features
        s.features = ["does_not_exist"]
        # ensure it stayed the same
        assert s.features == f_old

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
