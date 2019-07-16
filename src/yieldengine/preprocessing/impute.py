"""Impute and indicate missing values."""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import MissingIndicator, SimpleImputer

from yieldengine.df.transform import ColumnPreservingTransformer, DataFrameTransformer

log = logging.getLogger(__name__)

__all__ = ["SimpleImputerDF", "MissingIndicatorDF"]


class SimpleImputerDF(ColumnPreservingTransformer[SimpleImputer]):
    """Wrap sklearn `SimpleImputer` and return a DataFrame.

    The parameters are the same as the one passed to sklearn `SimpleImputer`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> SimpleImputer:
        return SimpleImputer(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        stats = self.base_transformer.statistics_
        if issubclass(stats.dtype.type, float):
            nan_mask = np.isnan(stats)
        else:
            nan_mask = [
                x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
            ]
        return self.columns_in.delete(np.argwhere(nan_mask))


class MissingIndicatorDF(DataFrameTransformer[MissingIndicator]):
    """Wrap sklearn `MissingIndicator` and returns a DataFrame.

    The parameters are the same as the one passed to sklearn `MissingIndicator`.
    """

    def __init__(
        self,
        missing_values=np.nan,
        features="missing-only",
        sparse="auto",
        error_on_new=True,
        **kwargs,
    ) -> None:
        super().__init__(
            missing_values=missing_values,
            features=features,
            sparse=sparse,
            error_on_new=error_on_new,
            **kwargs,
        )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> MissingIndicator:
        return MissingIndicator(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        columns_original: np.ndarray = self.columns_in[
            self.base_transformer.features_
        ].values
        columns_out = pd.Index([f"{name}__missing" for name in columns_original])
        return pd.Series(index=columns_out, data=columns_original)
