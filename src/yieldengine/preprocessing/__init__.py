import logging
from typing import *

import pandas as pd
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from yieldengine.df.transform import (
    _BaseTransformer,
    ColumnPreservingTransformer,
    constant_column_transformer,
    ConstantColumnTransformer,
)

log = logging.getLogger(__name__)


@constant_column_transformer(source_transformer=MinMaxScaler)
class MinMaxScalerDF(ConstantColumnTransformer[MinMaxScaler]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


@constant_column_transformer(source_transformer=StandardScaler)
class StandardScalerDF(ConstantColumnTransformer[StandardScaler]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


@constant_column_transformer(source_transformer=MaxAbsScaler)
class MaxAbsScalerDF(ConstantColumnTransformer[MaxAbsScaler]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


@constant_column_transformer(source_transformer=RobustScaler)
class RobustScalerDF(ConstantColumnTransformer[RobustScaler]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


@constant_column_transformer(source_transformer=PowerTransformer)
class PowerTransformerDF(ConstantColumnTransformer[PowerTransformer]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


@constant_column_transformer(source_transformer=QuantileTransformer)
class QuantileTransformerDF(ConstantColumnTransformer[QuantileTransformer]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


@constant_column_transformer(source_transformer=Normalizer)
class NormalizerDF(ConstantColumnTransformer[Normalizer]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass


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
