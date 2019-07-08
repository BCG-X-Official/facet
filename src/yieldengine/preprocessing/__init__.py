import logging
from typing import *

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
    make_constant_column_transformer_class,
)

log = logging.getLogger(__name__)


MaxAbsScalerDF = make_constant_column_transformer_class(
    base_transformer_type=MaxAbsScaler
)

MinMaxScalerDF = make_constant_column_transformer_class(
    base_transformer_type=MinMaxScaler
)

NormalizerDF = make_constant_column_transformer_class(base_transformer_type=Normalizer)

PowerTransformerDF = make_constant_column_transformer_class(
    base_transformer_type=PowerTransformer
)

QuantileTransformerDF = make_constant_column_transformer_class(
    base_transformer_type=QuantileTransformer
)

RobustScalerDF = make_constant_column_transformer_class(
    base_transformer_type=RobustScaler
)

StandardScalerDF = make_constant_column_transformer_class(
    base_transformer_type=StandardScaler
)

KernelCentererDF = make_constant_column_transformer_class(
    base_transformer_type=KernelCenterer
)


class FunctionTransformerDF(ColumnPreservingTransformer[FunctionTransformer]):
    def _get_columns_out(self) -> pd.Index:
        if isinstance(self._columns_out_provided, Callable):
            return self._columns_out_provided()
        else:
            # ignore error, type is already checked correctly in constructor:
            # noinspection PyTypeChecker
            return self._columns_out_provided

    def __init__(
        self, columns_out: Union[pd.Index, Callable[[None], pd.Index]], **kwargs
    ) -> None:

        if columns_out is None:
            raise ValueError("'columns_out' is required")

        if not isinstance(columns_out, pd.Index) and not isinstance(
            columns_out, Callable
        ):
            raise TypeError(
                "'columns_out' must be pandas.Index or callable->pandas.Index"
            )

        self._columns_out_provided = columns_out

        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> FunctionTransformer:
        return FunctionTransformer(**kwargs)


class PolynomialFeaturesDF(ColumnPreservingTransformer[PolynomialFeatures]):
    def _get_columns_out(self) -> pd.Index:
        return pd.Index(
            data=self.base_transformer.get_feature_names(input_features=self.columns_in)
        )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> PolynomialFeatures:
        return PolynomialFeatures(**kwargs)
