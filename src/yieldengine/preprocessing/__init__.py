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
    constant_column_transformer,
)

log = logging.getLogger(__name__)


@constant_column_transformer
class MaxAbsScalerDF(MaxAbsScaler):
    pass


@constant_column_transformer
class MinMaxScalerDF(MinMaxScaler):
    pass


@constant_column_transformer
class NormalizerDF(Normalizer):
    pass


@constant_column_transformer
class PowerTransformerDF(PowerTransformer):
    pass


@constant_column_transformer
class QuantileTransformerDF(QuantileTransformer):
    pass


@constant_column_transformer
class RobustScalerDF(RobustScaler):
    pass


@constant_column_transformer
class StandardScalerDF(StandardScaler):
    pass


@constant_column_transformer
class KernelCentererDF(KernelCenterer):
    pass


@constant_column_transformer
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
