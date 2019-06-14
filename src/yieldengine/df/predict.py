import logging
from abc import ABCMeta
from typing import Any, Generic, Optional, TypeVar, Union

import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from yieldengine.df import DataFrameEstimator

log = logging.getLogger(__name__)

_BasePredictor = TypeVar("_BasePredictor", bound=Union[RegressorMixin, ClassifierMixin])


class DataFramePredictor(
    Generic[_BasePredictor], DataFrameEstimator[_BasePredictor], metaclass=ABCMeta
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame, **predict_params):
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict(X, **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        self._check_parameter_types(X, y)

        # noinspection PyUnresolvedReferences
        result = self.base_estimator.fit_predict(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return result

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame):
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict_proba(X)

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame):
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self.base_estimator.decision_function(X)

    # noinspection PyPep8Naming
    def predict_log_proba(self, X: pd.DataFrame):
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
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
