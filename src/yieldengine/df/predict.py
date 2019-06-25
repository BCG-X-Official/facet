# coding=utf-8
"""Base classes for wrapper around regressor and classifier returning pandas objects."""

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
    """
    Wrapper around scikit-learn regressor and classifiers that preserves dataframes.
    """
    F_PREDICTION = "prediction"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame, **predict_params) -> pd.Series:
        """
        Returns prediction as a `pd.Series`.

        :param X: the dataframe of features
        :param predict_params: additional arguments passed to the the `predict` method \
        of the base estimator
        :return: the `pd.Series` of predictions
        """
        self._check_parameter_types(X, None)

        return self._prediction_to_series(X, self._base_predict(X, **predict_params))

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        """
        Fit and predict.
        :param X: the dataframe of features
        :param y: the series of target used to train the model
        :param fit_params: additional arguments passed to the the `predict` method
        of the base estimator
        :return: `pd.Series` of the predictions for X
        """
        self._check_parameter_types(X, y)

        result = self._prediction_to_series(
            X, self._base_fit_predict(X, y, **fit_params)
        )

        self._post_fit(X, y, **fit_params)

        return result

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Probability estimates.

        :param X: dataframe of features
        :return: the series of probabiliy estimates
        """
        self._check_parameter_types(X, None)

        return self._prediction_to_series(X, self._base_predict_proba(X))

    # noinspection PyPep8Naming
    def predict_log_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Log of probability estimates.

        :param X: dataframe of features
        :return: series of log-probabilities
        """
        self._check_parameter_types(X, None)

        return self._prediction_to_series(X, self._base_predict_log_proba(X))

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the decision function for the samples in X.
        :param X: dataframe of features
        :return: dataframe of the decision functions of the sample for each class
        """
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
        """
        Returns the score of the base estimator.

        :param X: dataframe of the features, shape = (n_samples, n_features)
        :param y: series of the true targets, shape = (n_samples) or (n_samples, \
        n_outputs)
        :param sample_weight:  array-like, sample weights, shape = (n_sample)
        :return: the score of the model
        """
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
    data frames, but only numpy ndarrays.
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
