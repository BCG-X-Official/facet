# coding=utf-8
"""Base classes for wrapper around regressors and classifiers returning dataframes."""

import logging
from abc import ABC
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from yieldengine import ListLike, MatrixLike
from yieldengine.df import DataFrameEstimator

log = logging.getLogger(__name__)

_BasePredictor = TypeVar("_BasePredictor", bound=Union[RegressorMixin, ClassifierMixin])
_BaseRegressor = TypeVar("_BaseRegressor", bound=RegressorMixin)
_BaseClassifier = TypeVar("_BaseClassifier", bound=ClassifierMixin)


class DataFramePredictor(DataFrameEstimator[_BasePredictor], ABC):
    """
    Base class for sklearn regressors and classifiers that preserve data frames

    :param `**kwargs`: arguments passed to `DataFrameEstimator` in `__init__`
    """

    F_PREDICTION = "prediction"

    @property
    def n_outputs(self) -> int:
        """
        Number of outputs predicted by this predictor.

        Defaults to 1 if base predictor does not define property `n_outputs_`.
        """
        if self.is_fitted:
            return getattr(self.base_estimator, "n_outputs_", 1)
        else:
            raise AttributeError("n_outputs not defined for unfitted predictor")

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the prediction as a series or a dataframe.

        For single-output problems, return a series, fro multi-output problems,
        return a dataframe.

        :param X: the data frame of features
        :param predict_params: additional arguments passed to the `predict` method \
        of the base estimator
        :return: the predictions
        """
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_to_series_or_frame(
            X, self.base_estimator.predict(X, **predict_params)
        )

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        """
        Fit and return the predictions.

        :param X: the data frame of features
        :param y: the series of target used to train the model
        :param fit_params: additional arguments passed to the the `predict` method
          of the base estimator
        :return: `pd.Series` of the predictions for X
        """
        self._check_parameter_types(X, y)

        # noinspection PyUnresolvedReferences
        result = self._prediction_to_series_or_frame(
            X, self.base_estimator.fit_predict(X, y, **fit_params)
        )

        self._post_fit(X, y, **fit_params)

        return result

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """
        Return the score of the base estimator.

        :param X: data frame of the features, shape = (n_samples, n_features)
        :param y: series of the true targets, shape = (n_samples) or (n_samples, \
        n_outputs)
        :param sample_weight:  array-like, sample weights, shape = (n_sample)
        :return: the score of the model
        """
        self._check_parameter_types(X, None)
        return self.base_estimator.score(X, y, sample_weight)

    # noinspection PyPep8Naming
    def _prediction_to_series_or_frame(
        self, X: pd.DataFrame, y: MatrixLike[Any]
    ) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            # if we already have a series or data frame, return it unchanged
            return y
        elif isinstance(y, np.ndarray):
            if len(y) == len(X):
                # predictions are usually provided as an ndarray of the same length as X
                if len(y.shape) == 1:
                    # single-output predictions yield an ndarray of shape (n_samples)
                    return pd.Series(data=y, name=self.F_PREDICTION, index=X.index)
                if len(y.shape) == 2:
                    # multi-output predictions yield an ndarray of shape (n_samples,
                    # n_outputs)
                    return pd.DataFrame(data=y, index=X.index)
            raise TypeError(
                f"Unexpected shape of ndarray returned as prediction:" f" {y.shape}"
            )
        raise TypeError(
            f"unexpected data type returned as prediction: " f"{type(y).__name__}"
        )


class DataFrameRegressor(DataFramePredictor[_BaseRegressor]):
    """
    Wrapper around sklearn regressors that preserves data frames.
    """


class DataFrameClassifier(DataFramePredictor[_BaseClassifier]):
    """
    Wrapper around sklearn classifiers that preserves data frames.
    """

    @property
    def classes(self) -> Optional[ListLike[Any]]:
        """
        Classes of this classifier after fitting.

        ``None`` if the base estimator has no `classes_` property.
        """
        if self.is_fitted:
            return getattr(self.base_estimator, "classes_", None)
        else:
            raise AttributeError("classes not defined for unfitted classifier")

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Probability estimates.

        :param X: data frame of features
        :return: the series of probability estimates
        """
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X, self.base_estimator.predict_proba(X)
        )

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Log of probability estimates.

        :param X: data frame of features
        :return: series of log-probabilities
        """
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X, self.base_estimator.predict_log_proba(X)
        )

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Evaluate the decision function for the samples in X.

        :param X: data frame of features
        :return: data frame of the decision functions of the sample for each class
        """
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_with_class_labels(
            X, self.base_estimator.decision_function(X)
        )

    # noinspection PyPep8Naming
    def _prediction_with_class_labels(
        self, X: pd.DataFrame, y: MatrixLike[Any]
    ) -> Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]:
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            # if we already have a series or data frame, return it unchanged
            return y
        elif isinstance(y, list) and self.n_outputs > 1:
            # if we have a multi-output classifier, prediction of probabilities
            # yields a list of ndarrays
            return [self._prediction_with_class_labels(X, output) for output in y]
        elif isinstance(y, np.ndarray):
            if len(y) == len(X):
                # predictions of probabilities are usually provided as an ndarray the
                # same length as X
                if len(y.shape) == 1:
                    # for a binary classifier, we get a series with probabilities for
                    # the second class
                    return pd.Series(data=y, index=X.index, name=self.classes[1])
                elif len(y.shape) == 2:
                    # for a multi-class classifiers, we get a two-dimensional ndarray
                    # with probabilities for each class
                    return pd.DataFrame(data=y, index=X.index, columns=self.classes)
            raise TypeError(
                f"Unexpected shape of ndarray returned as prediction: {y.shape}"
            )
        raise TypeError(f"unexpected type or prediction result: {type(y).__name__}")
