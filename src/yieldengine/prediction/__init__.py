import logging
from typing import Any, Optional, Union

import pandas as pd

from yieldengine.df.predict import DataFramePredictor

log = logging.getLogger(__name__)


class _DataFramePredictor(DataFramePredictor):
    """Dummy data frame predictor class, for type hinting only."""

    @property
    def n_outputs(self) -> int:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "DataFramePredictor":
        """Dummy implementation."""
        raise NotImplementedError()

    @property
    def is_fitted(self) -> bool:
        """Dummy implementation."""
        raise NotImplementedError()

    @property
    def columns_in(self) -> pd.Index:
        """Dummy implementation."""
        raise NotImplementedError()
