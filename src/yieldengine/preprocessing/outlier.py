import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


log = logging.getLogger(__name__)

__all__ = ["TukeyOutlierRemoverDF"]


class TukeyOutlierRemoverDF(BaseEstimator, TransformerMixin):
    """
    Remove outliers according to Tukey's method, respective to the interquartile \
    range (IQR)
    """

    def __init__(self, iqr_threshold: float):
        self.iqr_threshold = iqr_threshold

    def fit(self, X: pd.DataFrame, y=Optional[pd.Series]) -> None:
        """

        :param X:
        :param y:
        :return:
        """
        q1 = X.quantile(q=.25)
        q3 = X.quantile(q=.75)
        iqr = q3 - q1
        self.low = q1 - self.iqr_threshold * iqr
        self.high = q3 + self.iqr_threshold *iqr
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        # define a boolean mask of the outliers
        mask = (X < self.low) | (X > self.high)
        X_return = X.where(cond=~mask, other=np.nan)
        return X_return
