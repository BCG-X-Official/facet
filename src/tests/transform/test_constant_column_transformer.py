import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from yieldengine import Sample
from yieldengine.df.transform import (
    _BaseTransformer,
    constant_column_transformer,
    ConstantColumnTransformer,
)

log = logging.getLogger(__name__)


def test_make_constant_column_transformer(sample: Sample) -> None:
    @constant_column_transformer(source_transformer=MinMaxScaler)
    class MinMaxScalerDF(ConstantColumnTransformer[MinMaxScaler]):
        @classmethod
        def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
            pass

    scaler = MinMaxScalerDF()

    numerical_features = sample.features_by_type(Sample.DTYPE_NUMERICAL)

    transformed = scaler.fit_transform(X=numerical_features)

    # check for correct return type
    assert isinstance(transformed, pd.DataFrame)

    # check for correct columns
    assert np.all(transformed.columns == numerical_features.columns)

    # check scaling
    assert transformed.loc[:, transformed.columns[0]].min() < 0.00001
    assert transformed.loc[:, transformed.columns[0]].max() > 0.99999
