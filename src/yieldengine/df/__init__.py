# coding=utf-8
"""
Wrap sklearn `BaseEstimator` to return dataframes instead of numpy arrays.

The abstract class `~DataFrameEstimator` wraps a `BaseEstimator` so that the `predict`
or `transform` methods of the implementations return dataframe.
`~DataFrameEstimator` has an attribute `columns_in` which is the index of the
columns of the input dataframe.
"""

import logging
from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from sklearn.base import BaseEstimator

log = logging.getLogger(__name__)

_BaseEstimator = TypeVar("_BaseEstimator", bound=BaseEstimator)


class DataFrameEstimator(ABC, BaseEstimator, Generic[_BaseEstimator]):
    """
    Abstract base class that is a wrapper around the sklearn `BaseEstimator` class.

    Implementations must define a method `_make_base_estimator`.

    :param `**kwargs`: the arguments passed to the base estimator
    """

    F_COLUMN_IN = "column_in"

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
        Return the base sklearn `BaseEstimator`.

        :return: the estimator underlying self
        """
        return self._base_estimator

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: If True, return the parameters for this estimator and \
        contained sub-objects that are estimators

        :return: mapping of the parameter names to their values
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
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "DataFrameEstimator[_BaseEstimator]":
        """
        Fit the base estimator.

        :param X: dataframe to fit the estimator
        :param y: pandas series
        """
        self._check_parameter_types(X, y)

        self._base_fit(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self

    @property
    def is_fitted(self) -> bool:
        """``True`` if the base estimator is fitted, else ``False``."""
        return self._columns_in is not None

    @property
    def columns_in(self) -> pd.Index:
        """The index of the input columns."""
        self._ensure_fitted()
        return self._columns_in

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("transformer not fitted")

    # noinspection PyPep8Naming
    def _base_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> _BaseEstimator:
        # noinspection PyUnresolvedReferences
        return self._base_estimator.fit(X, y, **fit_params)

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> None:
        self._columns_in = X.columns.rename(self.F_COLUMN_IN)

    # noinspection PyPep8Naming
    @staticmethod
    def _check_parameter_types(X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be None or a Series")
