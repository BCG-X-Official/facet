import logging
from abc import ABCMeta
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from yieldengine.df import DataFrameEstimator, ListLike, MatrixLike

log = logging.getLogger(__name__)

_BasePredictor = TypeVar("_BasePredictor", bound=Union[RegressorMixin, ClassifierMixin])


class DataFramePredictor(DataFrameEstimator[_BasePredictor], metaclass=ABCMeta):
    F_PREDICTION = "prediction"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame, **predict_params) -> pd.Series:
        self._check_parameter_types(X, None)

        return self._prediction_to_series(X, self._base_predict(X, **predict_params))

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        self._check_parameter_types(X, y)

        result = self._prediction_to_series(
            X, self._base_fit_predict(X, y, **fit_params)
        )

        self._post_fit(X, y, **fit_params)

        return result

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        self._check_parameter_types(X, None)

        return self._prediction_to_series(X, self._base_predict_proba(X))

    # noinspection PyPep8Naming
    def predict_log_proba(self, X: pd.DataFrame) -> pd.Series:
        self._check_parameter_types(X, None)

        return self._prediction_to_series(X, self._base_predict_log_proba(X))

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_parameter_types(X, None)

        return pd.DataFrame(
            data=self._base_decision_function(X),
            index=X.index,
            columns=getattr(self.base_estimator, "classes_", None),
        )

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        self._check_parameter_types(X, None)
        return self._base_score(X, y, sample_weight)

    # noinspection PyPep8Naming
    def _prediction_to_series(
        self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series, Sequence[Any]]
    ) -> pd.Series:
        if isinstance(y, pd.Series):
            return y
        else:
            return pd.Series(y, name=self.F_PREDICTION, index=X.index)

    # noinspection PyPep8Naming
    def _base_predict(self, X: pd.DataFrame, **predict_params) -> ListLike:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict(X, **predict_params)

    # noinspection PyPep8Naming
    def _base_fit_predict(self, X: pd.DataFrame, y: ListLike, **fit_params) -> ListLike:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.fit_predict(X, y, **fit_params)

    # noinspection PyPep8Naming
    def _base_predict_proba(self, X: pd.DataFrame, **predict_params) -> ListLike[float]:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict_proba(X, **predict_params)

    # noinspection PyPep8Naming
    def _base_predict_log_proba(
        self, X: pd.DataFrame, **predict_params
    ) -> ListLike[float]:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict_log_proba(X, **predict_params)

    # noinspection PyPep8Naming
    def _base_decision_function(self, X: pd.DataFrame) -> MatrixLike[float]:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.decision_function(X)

    # noinspection PyPep8Naming
    def _base_score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series
    ) -> float:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.score(X, y, sample_weight)


class NDArrayPredictorDF(
    Generic[_BasePredictor], DataFramePredictor[_BasePredictor], metaclass=ABCMeta
):
    """
    Special case of DataFrameTransformer where the base transformer does not accept
    data frames, but only numpy ndarrays
    """

    # noinspection PyPep8Naming
    def _base_predict(self, X: pd.DataFrame, **predict_params) -> ListLike:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict(X.values, **predict_params)

    # noinspection PyPep8Naming
    def _base_fit_predict(self, X: pd.DataFrame, y: ListLike, **fit_params) -> ListLike:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.fit_predict(X.values, y.values, **fit_params)

    # noinspection PyPep8Naming
    def _base_predict_proba(self, X: pd.DataFrame, **predict_params) -> ListLike[float]:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict_proba(X.values, **predict_params)

    # noinspection PyPep8Naming
    def _base_predict_log_proba(
        self, X: pd.DataFrame, **predict_params
    ) -> ListLike[float]:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.predict_log_proba(X.values, **predict_params)

    # noinspection PyPep8Naming
    def _base_decision_function(self, X: pd.DataFrame) -> MatrixLike[float]:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.decision_function(X.values)

    # noinspection PyPep8Naming
    def _base_score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series
    ) -> float:
        # noinspection PyUnresolvedReferences
        return self.base_estimator.score(X.values, y.values, sample_weight.values)
