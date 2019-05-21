from typing import *

from yieldengine import Sample
from yieldengine.feature.transform import DataFrameTransformer


def test_column_transformer_df(
    sample: Sample, transformer_step: Tuple[str, DataFrameTransformer]
) -> None:
    transformer_step[1].fit_transform(X=sample.features)
