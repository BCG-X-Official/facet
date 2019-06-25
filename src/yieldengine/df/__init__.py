# coding=utf-8
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

log = logging.getLogger(__name__)

_BaseEstimator = TypeVar("_BaseEstimator", bound=BaseEstimator)
# noinspection PyShadowingBuiltins
_T = TypeVar("_T")

ListLike = Union[np.ndarray, pd.Series, Sequence[_T]]
MatrixLike = Union[np.ndarray, pd.DataFrame, Sequence[Sequence[_T]]]


class DataFrameEstimator(ABC, BaseEstimator, Generic[_BaseEstimator]):
    """
    Abstract base class that is a wrapper around the scikit-learn `BaseEstimator` class.
    """
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
        """
        Returns the base scikit-learn estimator.

        :return: the estimator underlying this DataFrameEstimator
        """
        return self._base_estimator

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: If True, will return the parameters for this estimator and \
        contained sub-objects that are estimators

        :return: params Parameter names mapped to their values
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

    # noinspection PyPep8Naming
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> None:
        """
        Fit the base estimator.

        :param X: dataframe to fit the estimator
        :param y: pandas series
        """
        self._check_parameter_types(X, y)

        self._base_fit(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

    @property
    def is_fitted(self) -> bool:
        """True if the base estimator is fitted, else false"""
        return self._columns_in is not None

    @property
    def columns_in(self) -> pd.Index:
        """The index of the input columns"""
        self._ensure_fitted()
        return self._columns_in

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("transformer not fitted")

    # noinspection PyPep8Naming
    def _base_fit(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params) -> None:
        # noinspection PyUnresolvedReferences
        self.base_transformer.fit(X, y, **fit_params)

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> None:
        self._columns_in = X.columns.rename(self.F_COLUMN)

    # noinspection PyPep8Naming
    @staticmethod
    def _check_parameter_types(X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be a Series")
