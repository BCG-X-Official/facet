# coding=utf-8
"""
Wrap scikit-learn `BaseEstimator` to return dataframes instead of numpy arrays.

The abstract class :class:`DataFrameEstimator` wraps
:class:`~sklearn.base.BaseEstimator` so that the ``predict``
and ``transform`` methods of the implementations return dataframe.
:class:`DataFrameEstimator` has an attribute :attr:`~DataFrameEstimator.columns_in`
which is the index of the columns of the input dataframe.
"""

import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import *

import pandas as pd
from sklearn.base import BaseEstimator

from yieldengine.df import T_BaseEstimator
from yieldengine.df.transform import T_BaseEstimator

log = logging.getLogger(__name__)

_BaseEstimator = TypeVar("_BaseEstimator", bound=BaseEstimator)


class DataFrameEstimator(ABC, BaseEstimator):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._columns_in = None

    # noinspection PyPep8Naming
    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "DataFrameEstimator":
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """`True` if the base estimator is fitted, else `False`"""
        pass

    @property
    @abstractmethod
    def columns_in(self) -> pd.Index:
        """The names of the input columns this estimator was fitted on"""
        pass


class DataFrameEstimatorWrapper(DataFrameEstimator, Generic[_BaseEstimator]):
    """
    Abstract base class that is a wrapper around :class:`sklearn.base.BaseEstimator`.

    Implementations must define a method ``_make_base_estimator``.

    :param `**kwargs`: the arguments passed to the base estimator
    """

    F_COLUMN_IN = "column_in"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._base_estimator = type(self)._make_base_estimator(**kwargs)

    @classmethod
    @abstractmethod
    def _make_base_estimator(cls, **kwargs) -> _BaseEstimator:
        pass

    @property
    def base_estimator(self) -> _BaseEstimator:
        """
        Return the base sklearn `BaseEstimator`.

        :return: the estimator underlying this DataFrameEstimatorWrapper
        """
        return self._base_estimator

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator and \
        contained sub-objects that are estimators

        :return: mapping of the parameter names to their values
        """
        # noinspection PyUnresolvedReferences
        return self._base_estimator.get_params(deep=deep)

    def set_params(self, **kwargs) -> "DataFrameEstimatorWrapper":
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
    ) -> "DataFrameEstimatorWrapper[_BaseEstimator]":
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

    def __dir__(self) -> Iterable[str]:
        return {*super().__dir__(), *self._base_estimator.__dir__()}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        else:
            return getattr(self._base_estimator, name)

    def __setattr__(self, name: str, value: Any) -> Any:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._base_estimator, name, value)


def df_estimator(
    base_estimator: Type[T_BaseEstimator] = None,
    *,
    df_estimator_type: Type[
        DataFrameEstimatorWrapper[T_BaseEstimator]
    ] = DataFrameEstimatorWrapper[T_BaseEstimator],
) -> Union[
    Callable[[Type[T_BaseEstimator]], Type[DataFrameEstimatorWrapper[T_BaseEstimator]]],
    Type[DataFrameEstimatorWrapper[T_BaseEstimator]],
]:
    """
    Class decorator wrapping a :class:`sklearn.base.BaseEstimattor` in a
    :class:`DataFrameEstimatorWrapper`.
    :param base_estimator: the estimator class to wrap
    :param df_estimator_type: optional parameter indicating the \
                              :class:`DataFrameEstimatorWrapper` class to be used for \
                              wrapping; defaults to :class:`DataFrameEstimatorWrapper`
    :return: the resulting `DataFrameEstimatorWrapper` with ``base_estimator`` as \
             the base estimator
    """

    def _decorate(
        decoratee: Type[T_BaseEstimator]
    ) -> Type[DataFrameEstimatorWrapper[T_BaseEstimator]]:
        @wraps(decoratee, updated=())
        class _DataFrameEstimator(df_estimator_type):
            @classmethod
            def _make_base_estimator(cls, **kwargs) -> T_BaseEstimator:
                return decoratee(**kwargs)

        decoratee.__name__ = f"_{decoratee.__name__}Base"
        decoratee.__qualname__ = f"{decoratee.__qualname__}.{decoratee.__name__}"
        setattr(_DataFrameEstimator, decoratee.__name__, decoratee)
        return _DataFrameEstimator

    if not issubclass(df_estimator_type, DataFrameEstimatorWrapper):
        raise ValueError(
            f"arg df_transformer_type not a "
            f"{DataFrameEstimatorWrapper.__name__} class: {df_estimator_type}"
        )
    if base_estimator is None:
        return _decorate
    else:
        return _decorate(base_estimator)
