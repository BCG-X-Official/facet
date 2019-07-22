import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from gamma.sklearndf.transformation import (
    ColumnTransformerDF,
    MaxAbsScalerDF,
    MinMaxScalerDF,
    NormalizerDF,
    PowerTransformerDF,
    QuantileTransformerDF,
    RobustScalerDF,
    StandardScalerDF,
)
from gamma.sklearndf.transformation.extra import OutlierRemoverDF


@pytest.fixture
def test_data() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c0": [0, 1, 2.5, 3, 4, 5.2, 6, 7, 8, 9],
            "c1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )


def test_various(test_data: pd.DataFrame) -> None:

    to_test = [
        (StandardScalerDF, StandardScaler),
        (MinMaxScalerDF, MinMaxScaler),
        (MaxAbsScalerDF, MaxAbsScaler),
        (RobustScalerDF, RobustScaler),
        (PowerTransformerDF, PowerTransformer),
        (QuantileTransformerDF, QuantileTransformer),
    ]

    for df_transf, src_transf in to_test:
        # initalize both kind of transformers
        df_t = df_transf()
        non_df_t = src_transf()

        # test fit-transform on both in conjecture with ColumnTransformer(DF)
        df_col_t = ColumnTransformerDF(
            transformers=[("t", df_t, ["c0"])], remainder="drop"
        )
        transformed_df = df_col_t.fit_transform(X=test_data)

        assert isinstance(transformed_df, pd.DataFrame)

        non_df_col_t = ColumnTransformer(transformers=[("t", non_df_t, ["c0"])])

        transformed_non_df = non_df_col_t.fit_transform(X=test_data)

        assert "c0" in transformed_df.columns
        assert np.all(
            np.round(transformed_df["c0"].values, 1)
            == np.round(transformed_non_df.reshape(10), 1)
        )


def test_normalizer_df() -> None:
    x = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
    test_df = pd.DataFrame(x)
    test_df.columns = ["a", "b", "c", "d"]

    non_df_normalizer = Normalizer(norm="l2")
    df_normalizer = NormalizerDF(norm="l2")

    transformed_non_df = non_df_normalizer.fit_transform(X=x)
    transformed_df = df_normalizer.fit_transform(X=test_df)

    # check equal results:
    assert np.array_equal(transformed_non_df, transformed_df.values)
    # check columns are preserved:
    assert np.all(transformed_df.columns == ["a", "b", "c", "d"])


@pytest.fixture
def df_outlier() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "c0": [0, 1, 2, 3, 4],
            "c1": [-1, 0, 0, 0, 1],
            "c2": [-10, 0, 1, 2, 3],
            "c3": [0, 1, 2, 3, 10],
        }
    )


def test_outlier_remover(df_outlier: pd.DataFrame) -> None:
    outlier_remover = OutlierRemoverDF(iqr_multiple=2)
    df_transformed = outlier_remover.fit_transform(df_outlier)
    df_transformed_expected = pd.DataFrame(
        data={
            "c0": [0, 1, 2, 3, 4],
            "c1": [np.nan, 0, 0, 0, np.nan],
            "c2": [np.nan, 0, 1, 2, 3],
            "c3": [0, 1, 2, 3, np.nan],
        }
    )
    assert_frame_equal(df_transformed, df_transformed_expected)
