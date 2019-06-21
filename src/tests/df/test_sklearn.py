import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from yieldengine.df.sklearn import (
    MaxAbsScalerDF,
    MinMaxScalerDF,
    RobustScalerDF,
    StandardScalerDF,
)
from yieldengine.preprocessing.compose import ColumnTransformerDF


def test_df_scalers() -> None:

    to_test = [
        (StandardScalerDF, StandardScaler),
        (MinMaxScalerDF, MinMaxScaler),
        (MaxAbsScalerDF, MaxAbsScaler),
        (RobustScalerDF, RobustScaler),
    ]

    data_to_test = pd.DataFrame(
        data={
            "c0": [0, 1, 2.5, 3, 4, 5.2, 6, 7, 8, 9],
            "c1": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )

    for df_transf, src_transf in to_test:
        # initalize both kind of transformers
        df_t = df_transf()
        non_df_t = src_transf()

        # test fit-transform on both using Column-Transformer
        df_col_t = ColumnTransformerDF(
            transformers=[("scale", df_t, ["c0"])], remainder="drop"
        )
        transformed_df = df_col_t.fit_transform(X=data_to_test)

        non_df_col_t = ColumnTransformer(transformers=[("scale", non_df_t, ["c0"])])

        transformed_non_df = non_df_col_t.fit_transform(X=data_to_test)

        assert "c0" in transformed_df.columns
        assert np.all(
            np.round(transformed_df["c0"].values, 1)
            == np.round(transformed_non_df.reshape(10), 1)
        )
