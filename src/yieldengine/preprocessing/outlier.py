"""
This module defines a transformers to remove outliers.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype

from yieldengine.df.transform import DataFrameTransformer, ConstantColumnTransformer

log = logging.getLogger(__name__)

__all__ = ["TukeyOutlierRemover", "TukeyOutlierRemoverDF"]


class TukeyOutlierRemover(BaseEstimator, TransformerMixin):
    """
    Transformer to remove outliers according to Tukey's method, respective to the
    interquartile \
    range (IQR)
    """

    def __init__(self, iqr_threshold: float):
        self.iqr_threshold = iqr_threshold

    def fit(self, X: pd.DataFrame, y=Optional[pd.Series]) -> "TukeyOutlierRemover":
        """
        Fit the transformer on X.

        :param X: input dataframe
        :param y: optional, target series
        :return: self, the fitted tranformer
        """
        if not all(X.apply(is_numeric_dtype)):
            raise ValueError("Non numerical dtype in X.")
        q1 = X.quantile(q=.25)
        q3 = X.quantile(q=.75)
        iqr = q3 - q1
        self.low = q1 - self.iqr_threshold * iqr
        self.high = q3 + self.iqr_threshold *iqr
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace outliers by nan.

        :param X: dataframe to tranform
        :return: transformed dataframe where outliers have been replaced by nan
        """
        if not all(X.apply(is_numeric_dtype)):
            raise ValueError("Non numerical dtype in X.")
        # define a boolean mask of the outliers
        mask = (X < self.low) | (X > self.high)
        X_return = np.where(~mask, X, np.nan)
        return X_return


class TukeyOutlierRemoverDF(ConstantColumnTransformer[TukeyOutlierRemover]):
    """
    Remove outliers according to Tukey's method.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> TukeyOutlierRemover:
        return TukeyOutlierRemover(**kwargs)
