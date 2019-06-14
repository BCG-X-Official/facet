import logging
from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from sklearn.base import BaseEstimator

log = logging.getLogger(__name__)

_BaseEstimator = TypeVar("_BaseEstimator", bound=BaseEstimator)


class DataFrameEstimator(ABC, BaseEstimator, Generic[_BaseEstimator]):
    F_COLUMN = "column"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._base_estimator = type(self)._make_base_estimator(**kwargs)
        self._columns_in = None

    @classmethod
    @abstractmethod
    def _make_base_estimator(cls, **kwargs) -> _BaseEstimator:
        pass

    @property
    def base_estimator(self) -> _BaseEstimator:
        return self._base_estimator

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep If True, will return the parameters for this estimator and
        contained subobjects that are estimators

        :returns params Parameter names mapped to their values
        """
        # noinspection PyUnresolvedReferences
        return self._base_estimator.get_params(deep=deep)

    def set_params(self, **kwargs) -> "DataFrameEstimator":
        """
        Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        :returns self
        """
        # noinspection PyUnresolvedReferences
        self._base_estimator.set_params(**kwargs)
        return self

    def is_fitted(self) -> bool:
        return self._columns_in is not None

    def _ensure_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError("transformer not fitted")

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> None:
        self._columns_in = X.columns.rename(self.F_COLUMN)

    @property
    def columns_in(self) -> pd.Index:
        self._ensure_fitted()
        return self._columns_in

    # noinspection PyPep8Naming
    @staticmethod
    def _check_parameter_types(X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be a Series")
