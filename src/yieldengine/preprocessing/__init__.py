"""
This module contains wrapper classes around sklearn preprocessing classes (
OneHotEncoder, Imputer, ...) that return dataframes with the correct row and column
names.
"""

import logging

from sklearn.preprocessing import FunctionTransformer

from yieldengine.df.transform import ConstantColumnTransformer

log = logging.getLogger(__name__)


class FunctionTransformerDF(ConstantColumnTransformer[FunctionTransformer]):
    """Wrapper around sklearn ```FunctionTransformer``` that returns a DataFrame
    with correct row and column indices."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> FunctionTransformer:
        return FunctionTransformer(**kwargs)
