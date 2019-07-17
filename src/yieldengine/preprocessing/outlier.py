"""Remove outliers."""

import logging
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from yieldengine.df.transform import df_transformer

log = logging.getLogger(__name__)


@df_transformer
class OutlierRemoverDF(BaseEstimator, TransformerMixin):
    """Remove outliers according to Tukey's method.

    A sample is considered an outlier if it is outside the range
    :math:`[Q_1 - iqr\\_ multiple(Q_3-Q_1), Q_3 + iqr\\_ multiple(Q_3-Q_1)]`
    where :math:`Q_1` and :math:`Q_3` are the lower and upper quartiles.

    :param float iqr_multiple: the multiple used to define the range of non-outlier
      samples in the above explanation
    """

    def __init__(self, iqr_multiple: float):
        if iqr_multiple < 0.0:
            raise ValueError(f"arg iqr_multiple is negative: {iqr_multiple}")
        self.iqr_multiple = iqr_multiple
        self.threshold_low_ = None
        self.threshold_high_ = None

    # noinspection PyPep8Naming
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierRemoverDF":
        """Fit the transformer."""
        q1: pd.Series = X.quantile(q=0.25)
        q3: pd.Series = X.quantile(q=0.75)
        threshold_iqr: pd.Series = (q3 - q1) * self.iqr_multiple
        self.threshold_low_ = q1 - threshold_iqr
        self.threshold_high_ = q3 + threshold_iqr
        return self

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X where outliers are replaced by Nan.

        :return: the dataframe X where outliers are replaced by Nan
        """
        return X.where(cond=(X >= self.threshold_low_) & (X <= self.threshold_high_))
