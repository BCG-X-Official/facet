import logging

import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    KernelCenterer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from yieldengine.df.transform import (
    ColumnPreservingTransformer,
    make_constant_column_transformer_type,
)

log = logging.getLogger(__name__)

MaxAbsScalerDF = make_constant_column_transformer_type(MaxAbsScaler)
MinMaxScalerDF = make_constant_column_transformer_type(MinMaxScaler)
NormalizerDF = make_constant_column_transformer_type(Normalizer)
PowerTransformerDF = make_constant_column_transformer_type(PowerTransformer)
QuantileTransformerDF = make_constant_column_transformer_type(QuantileTransformer)
RobustScalerDF = make_constant_column_transformer_type(RobustScaler)
StandardScalerDF = make_constant_column_transformer_type(StandardScaler)
KernelCentererDF = make_constant_column_transformer_type(KernelCenterer)
FunctionTransformerDF = make_constant_column_transformer_type(FunctionTransformer)


class PolynomialFeaturesDF(ColumnPreservingTransformer[PolynomialFeatures]):
    def _get_columns_out(self) -> pd.Index:
        return pd.Index(
            data=self.base_transformer.get_feature_names(input_features=self.columns_in)
        )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> PolynomialFeatures:
        return PolynomialFeatures(**kwargs)
