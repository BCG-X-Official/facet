import logging

from sklearn.preprocessing import FunctionTransformer

from yieldengine.df.transform import ConstantColumnTransformer

log = logging.getLogger(__name__)


class FunctionTransformerDF(ConstantColumnTransformer[FunctionTransformer]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> FunctionTransformer:
        return FunctionTransformer(**kwargs)
