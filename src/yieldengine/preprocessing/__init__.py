"""
Preprocessing
"""

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

from yieldengine.df.transform import ColumnPreservingTransformer, df_transformer

log = logging.getLogger(__name__)


@df_transformer
class MaxAbsScalerDF(MaxAbsScaler):
    pass


@df_transformer
class MinMaxScalerDF(MinMaxScaler):
    pass


@df_transformer
class NormalizerDF(Normalizer):
    pass


@df_transformer
class PowerTransformerDF(PowerTransformer):
    pass


@df_transformer
class QuantileTransformerDF(QuantileTransformer):
    pass


@df_transformer
class RobustScalerDF(RobustScaler):
    pass


@df_transformer
class StandardScalerDF(StandardScaler):
    pass


@df_transformer
class KernelCentererDF(KernelCenterer):
    pass


@df_transformer
class FunctionTransformerDF(FunctionTransformer):
    pass


class PolynomialFeaturesDF(ColumnPreservingTransformer[PolynomialFeatures]):
    def _get_columns_out(self) -> pd.Index:
        return pd.Index(
            data=self.base_transformer.get_feature_names(input_features=self.columns_in)
        )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> PolynomialFeatures:
        return PolynomialFeatures(**kwargs)
