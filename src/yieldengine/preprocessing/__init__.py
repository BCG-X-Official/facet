import logging

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


class FunctionTransformerDF(ConstantColumnTransformer[FunctionTransformer]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> FunctionTransformer:
        return FunctionTransformer(**kwargs)
