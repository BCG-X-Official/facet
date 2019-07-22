from typing import Sequence

from yieldengine.sklearndf import DataFrameTransformer
from yieldengine.sklearndf.transformation import (
    ColumnTransformerDF,
    OneHotEncoderDF,
    SimpleImputerDF,
)

STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


def make_simple_transformer(
    impute_median_columns: Sequence[str] = None,
    one_hot_encode_columns: Sequence[str] = None,
) -> DataFrameTransformer:
    column_transforms = []

    if impute_median_columns is not None and len(impute_median_columns) > 0:
        column_transforms.append(
            (STEP_IMPUTE, SimpleImputerDF(strategy="median"), impute_median_columns)
        )

    if one_hot_encode_columns is not None and len(one_hot_encode_columns) > 0:
        column_transforms.append(
            (
                STEP_ONE_HOT_ENCODE,
                OneHotEncoderDF(sparse=False, handle_unknown="ignore"),
                one_hot_encode_columns,
            )
        )

    return ColumnTransformerDF(transformers=column_transforms)
