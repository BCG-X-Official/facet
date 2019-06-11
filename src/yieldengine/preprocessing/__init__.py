import logging

import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from yieldengine.transform import ConstantColumnTransformer, DataFrameTransformer

log = logging.getLogger(__name__)


class OneHotEncoderDF(DataFrameTransformer[OneHotEncoder]):
    """
    A one-hot encoder that returns a DataFrame with correct row and column indices
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.base_transformer.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> OneHotEncoder:
        return OneHotEncoder(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        return pd.Series(
            index=pd.Index(self.base_transformer.get_feature_names(self.columns_in)),
            data=[
                column_original
                for column_original, category in zip(
                    self.columns_in, self.base_transformer.categories_
                )
                for _ in category
            ],
        )


class OrdinalEncoderDF(ConstantColumnTransformer[OrdinalEncoder]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> OrdinalEncoder:
        return OrdinalEncoder(**kwargs)


class FunctionTransformerDF(ConstantColumnTransformer[FunctionTransformer]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> FunctionTransformer:
        return FunctionTransformer(**kwargs)
