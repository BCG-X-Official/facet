from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from yieldengine.df.transform import (
    _BaseTransformer,
    constant_column_transformer,
    ConstantColumnTransformer,
)


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
