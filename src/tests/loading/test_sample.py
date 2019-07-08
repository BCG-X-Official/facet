import pandas as pd
import pytest
from joblib import delayed, Parallel

from yieldengine import Sample


# checks various erroneous inputs
def test_sample_init(batch_table: pd.DataFrame) -> None:
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
        Sample(observations=batch_table, target_name=None)

    # store list of feature columns:
    f_columns = list(batch_table.columns)
    f_columns.remove("Yield")

    # 2.1 invalid feature column specified
    with pytest.raises(ValueError):
        f_columns_false = f_columns.copy()
        f_columns_false.append("doesnt_exist")
        Sample(
            observations=batch_table, feature_names=f_columns_false, target_name="Yield"
        )

    # 2.2 invalid target column specified
    with pytest.raises(ValueError):
        Sample(
            observations=batch_table,
            target_name="doesnt_exist",
            feature_names=f_columns,
        )

    # 3. column is target and also feature
    with pytest.raises(ValueError):
        f_columns_false = f_columns.copy()
        f_columns_false.append("Yield")

        Sample(
            observations=batch_table, feature_names=f_columns_false, target_name="Yield"
        )


def test_sample(batch_table: pd.DataFrame) -> None:
    # define various assertions we want to test:
    def run_assertions(s: Sample):
        assert s.target_name == "Yield"
        assert "Yield" not in s.feature_names
        assert len(s.feature_names) == len(batch_table.columns) - 1

        assert type(s.target) == pd.Series
        assert type(s.features) == pd.DataFrame

        assert len(s.target) == len(s.features)

    # test explicit setting of both target & features
    feature_columns = list(batch_table.drop(columns="Yield").columns)
    s = Sample(
        observations=batch_table, target_name="Yield", feature_names=feature_columns
    )

    # run the checks on s:
    run_assertions(s)

    # test implicit setting of features by only giving the target
    s2 = Sample(observations=batch_table, target_name="Yield")

    # run the checks on s2:
    run_assertions(s2)

    # test numerical features
    features_numerical = s.features_by_type(dtype=s.DTYPE_NUMERICAL).columns
    assert (
        "Step4 Fermentation Sensor Data Phase2 Pressure Val04 (mbar)"
        in features_numerical
    )

    # test categorical features
    features_non_numerical = s.features_by_type(dtype=s.DTYPE_OBJECT).columns
    assert "Step4 RawMat Internal Compound01 QC (id)" in features_non_numerical

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
    assert len(s) == len(batch_table)

    # test select_observations
    sub = s2.select_observations(numbers=[0, 1, 2, 3])
    assert len(sub) == 4

    # test that s.features is a deterministic operation that does not depend on the
    # global python environment variable PYTHONHASHSEED
    parallel = Parallel(n_jobs=-3)

    def get_column(sample: Sample):
        return list(sample.features.columns)

    columns1, columns2 = parallel(delayed(get_column)(sample) for sample in [s, s])
    assert columns1 == columns2
