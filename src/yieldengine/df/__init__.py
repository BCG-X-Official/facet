import logging
from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

log = logging.getLogger(__name__)

_BaseEstimator = TypeVar("_BaseEstimator", bound=BaseEstimator)
_BasePredictor = TypeVar("_BasePredictor", bound=Union[RegressorMixin, ClassifierMixin])


class DataFrameEstimator(ABC, BaseEstimator, Generic[_BaseEstimator]):
    F_COLUMN_ORIGINAL = "column_original"
    F_COLUMN = "column"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._base_estimator = type(self)._make_base_estimator(**kwargs)
        self._columns_in = None
        self._columns_out = None
        self._columns_original = None

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
        self._columns_in = X.columns.rename(DataFrameEstimator.F_COLUMN)
        self._columns_out = None
        self._columns_original = None

    @abstractmethod
    def _get_columns_original(self) -> pd.Series:
        """
        :return: a mapping from this transformer's output columns to the original
        columns as a series
        """
        pass

    @property
    def columns_in(self) -> pd.Index:
        self._ensure_fitted()
        return self._columns_in

    @property
    def columns_out(self) -> pd.Index:
        return self.columns_original.index

    @property
    def columns_original(self) -> pd.Series:
        self._ensure_fitted()
        if self._columns_original is None:
            self._columns_original = (
                self._get_columns_original()
                .rename(DataFrameEstimator.F_COLUMN_ORIGINAL)
                .rename_axis(index=DataFrameEstimator.F_COLUMN)
            )
        return self._columns_original

    # noinspection PyPep8Naming
    @staticmethod
    def _check_parameter_types(X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be a Series")


class DataFramePredictor(DataFrameEstimator, ABC, Generic[_BasePredictor]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame, **predict_params):
        self._check_parameter_types(X, None)

        return self.base_estimator.predict(X, **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        self._check_parameter_types(X, y)

        result = self.base_estimator.fit_predict(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return result

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame):
        self._check_parameter_types(X, None)

        return self.base_estimator.predict_proba(X)

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame):
        self._check_parameter_types(X, None)

        return self.base_estimator.decision_function(X)

    # noinspection PyPep8Naming
    def predict_log_proba(self, X: pd.DataFrame):
        self._check_parameter_types(X, None)

        return self.base_estimator.predict_log_proba(X)

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ):
        self._check_parameter_types(X, None)
        return self.base_estimator.score(X, y, sample_weight)
