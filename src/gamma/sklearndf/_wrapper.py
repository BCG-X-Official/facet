"""
Wrappers around scikit-learn estimators.

These mimic the behavior of the wrapped scikit-learn estimator, but only accept and
return data frames (while scikit-learn transformers usually return a numpy arrays, and
may not accept data frames as input).

The wrappers also support the additional column attributes introduced by the
DataFrameEstimators and their generic subclasses including transformers and predictors
"""

import logging
from abc import ABC, abstractmethod
from functools import wraps
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from gamma import ListLike, MatrixLike
from gamma.sklearndf import (
    BaseEstimatorDF,
    BasePredictorDF,
    ClassifierDF,
    RegressorDF,
    T_Classifier,
    T_Estimator,
    T_Predictor,
    T_Regressor,
    T_Transformer,
    TransformerDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "BaseEstimatorWrapperDF",
    "BasePredictorWrapperDF",
    "RegressorWrapperDF",
    "ClassifierWrapperDF",
    "TransformerWrapperDF",
    "df_estimator",
]
#
# base wrapper classes
#


class BaseEstimatorWrapperDF(
    BaseEstimatorDF[T_Estimator], BaseEstimator, Generic[T_Estimator]
):
    """
    Abstract base class that is a wrapper around :class:`sklearn.base.BaseEstimator`.

    Implementations must define a method ``_make_delegate_estimator``.

    :param `**kwargs`: the arguments passed to the delegate estimator
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self._columns_in = None
        self._delegate_estimator = type(self)._make_delegate_estimator(*args, **kwargs)

    @classmethod
    @abstractmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> T_Estimator:
        pass

    @property
    def delegate_estimator(self) -> T_Estimator:
        """
        Return the original estimator which this wrapper delegates to.

        :return: the original estimator which this estimator delegates to
        """
        return self._delegate_estimator

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep: if ``True``, return the parameters for this estimator and \
        contained sub-objects that are estimators

        :return: mapping of the parameter names to their values
        """
        # noinspection PyUnresolvedReferences
        return self._delegate_estimator.get_params(deep=deep)

    def set_params(self, **kwargs) -> "BaseEstimatorWrapperDF[T_Estimator]":
        """
        Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        :returns self
        """
        # noinspection PyUnresolvedReferences
        self._delegate_estimator.set_params(**kwargs)
        return self

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "BaseEstimatorWrapperDF[T_Estimator]":
        """
        Fit the delegate estimator.

        :param X: data frame to fit the estimator
        :param y: pandas series
        """
        self._reset_fit()

        self._check_parameter_types(X, y)

        self._fit(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self

    @property
    def is_fitted(self) -> bool:
        """``True`` if this estimator is fitted, else ``False``."""
        return self._columns_in is not None

    def _get_columns_in(self) -> pd.Index:
        return self._columns_in

    def _reset_fit(self) -> None:
        self._columns_in = None

    # noinspection PyPep8Naming
    def _fit(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> T_Estimator:
        # noinspection PyUnresolvedReferences
        return self._delegate_estimator.fit(X, y, **fit_params)

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> None:
        self._columns_in = X.columns.rename(self.F_COLUMN_IN)

    # noinspection PyPep8Naming
    def _check_parameter_types(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if self.is_fitted:
            BaseEstimatorWrapperDF._verify_df(df=X, expected_columns=self.columns_in)
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be None or a Series")

    @staticmethod
    def _verify_df(
        df: pd.DataFrame, expected_columns: pd.Index, expected_index: pd.Index = None
    ) -> None:
        def _error_message(axis: str, actual: pd.Index, expected: pd.Index):
            error_message = f"transformed data frame does not have expected {axis}"
            missing_columns = expected.difference(actual)
            extra_columns = actual.difference(expected)
            error_detail = []
            if len(actual) != len(expected):
                error_detail.append(
                    f"expected {len(expected)} items but got {len(actual)}"
                )
            if len(missing_columns) > 0:
                error_detail.append(
                    f"missing columns: "
                    f"{', '.join(str(item) for item in missing_columns)}"
                )
            if len(extra_columns) > 0:
                error_detail.append(
                    f"extra columns: "
                    f"{', '.join(str(item) for item in extra_columns)}"
                )
            if len(error_detail) == 0:
                error_detail = [f"{axis} not in expected order"]
            return f"{error_message} ({'; '.join(error_detail)})"

        if not df.columns.equals(expected_columns):
            raise ValueError(
                _error_message(
                    axis="columns", actual=df.columns, expected=expected_columns
                )
            )
        if expected_index is not None and not df.index.equals(expected_index):
            raise ValueError(
                _error_message(axis="index", actual=df.index, expected=expected_index)
            )

    def __dir__(self) -> Iterable[str]:
        # include non-private attributes of delegate estimator in directory
        return {
            *super().__dir__(),
            *(
                attr
                for attr in self._delegate_estimator.__dir__()
                if not attr.startswith("_")
            ),
        }

    def __getattr__(self, name: str) -> Any:
        # get a public attribute of the delegate estimator
        if name.startswith("_"):
            raise AttributeError(name)
        else:
            return getattr(self._delegate_estimator, name)

    def __setattr__(self, name: str, value: Any) -> Any:
        # set a public attribute of the delegate estimator
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            setattr(self._delegate_estimator, name, value)


class TransformerWrapperDF(
    TransformerDF[T_Transformer],
    BaseEstimatorWrapperDF[T_Transformer],
    Generic[T_Transformer],
    ABC,
):
    """
    Wraps a :class:`sklearn.base.TransformerMixin` and ensures that the X and y
    objects passed and returned are pandas data frames with valid column names.

    Implementations must define ``_make_delegate_estimator`` and
    ``_get_columns_original``.

    :param `**args`: positional arguments of scikit-learn transformer to be wrapped
    :param `**kwargs`: keyword arguments  of scikit-learn transformer to be wrapped
    """

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Call the transform method of the delegate transformer
        ``self.delegate_estimator``.

        :param X: data frame to transform
        :return: transformed data frame
        """
        self._check_parameter_types(X, None)

        transformed = self._transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        """Call the ``fit_transform`` method of ``self.delegate_estimator``.

        :param X: data frame to transform
        :param y: series of training targets
        :param fit_params: parameters passed to the fit method of the delegate
                           transformer
        :return: data frame of transformed sample
        """
        self._reset_fit()

        self._check_parameter_types(X, y)

        transformed = self._fit_transform(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inverse transformations in reverse order on the delegate
        transformer.

        All estimators in the pipeline must support ``inverse_transform``.
        :param X: data frame of samples
        :return: data frame of inverse-transformed samples
        """
        self._reset_fit()

        self._check_parameter_types(X, None)

        transformed = self._inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_in
        )

    def _reset_fit(self) -> None:
        try:
            # noinspection PyProtectedMember
            super()._reset_fit()
        finally:
            self._columns_original = None

    @staticmethod
    def _transformed_to_df(
        transformed: Union[pd.DataFrame, np.ndarray], index: pd.Index, columns: pd.Index
    ):
        if isinstance(transformed, pd.DataFrame):
            # noinspection PyProtectedMember
            TransformerWrapperDF._verify_df(
                df=transformed, expected_columns=columns, expected_index=index
            )
            return transformed
        else:
            return pd.DataFrame(data=transformed, index=index, columns=columns)

    # noinspection PyPep8Naming
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.transform(X)

    # noinspection PyPep8Naming
    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.delegate_estimator.fit_transform(X, y, **fit_params)

    # noinspection PyPep8Naming
    def _inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.inverse_transform(X)


class BasePredictorWrapperDF(
    BasePredictorDF[T_Predictor],
    BaseEstimatorWrapperDF[T_Predictor],
    Generic[T_Predictor],
    ABC,
):
    """
    Base class for sklearn regressors and classifiers that preserve data frames

    :param `**kwargs`: arguments passed to :class:`.BaseEstimatorDF` in ``__init__``
    """

    F_PREDICTION = "prediction"

    @classmethod
    def from_fitted(
        cls: "Type[BasePredictorWrapperDF[T_Predictor]]",
        predictor: T_Predictor,
        columns_in: pd.Index,
    ) -> "BasePredictorWrapperDF[T_Predictor]":
        class _FittedPredictor(cls):
            def __init__(self) -> None:
                super().__init__()
                self._columns_in = columns_in

            @classmethod
            def _make_delegate_estimator(cls, *args, **kwargs) -> T_Predictor:
                return predictor

        return _FittedPredictor()

    @property
    def n_outputs(self) -> int:
        """
        Number of outputs predicted by this predictor.

        Defaults to 1 if base predictor does not define property ``n_outputs_``.
        """
        if self.is_fitted:
            return getattr(self.delegate_estimator, "n_outputs_", 1)
        else:
            raise AttributeError("n_outputs not defined for unfitted predictor")

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute the prediction as a series or a data frame.

        For single-output problems, return a series, fro multi-output problems,
        return a data frame.

        :param X: the data frame of features
        :param predict_params: additional arguments passed to the ``predict`` method \
        of the delegate estimator
        :return: the predictions
        """
        self._check_parameter_types(X, None)

        # noinspection PyUnresolvedReferences
        return self._prediction_to_series_or_frame(
            X, self.delegate_estimator.predict(X, **predict_params)
        )

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        """
        Fit and return the predictions.

        :param X: the data frame of features
        :param y: the series of target used to train the model
        :param fit_params: additional arguments passed to the the ``predict`` method
          of the delegate estimator
        :return: series of the predictions for X
        """
        self._check_parameter_types(X, y)

        # noinspection PyUnresolvedReferences
        result = self._prediction_to_series_or_frame(
            X, self.delegate_estimator.fit_predict(X, y, **fit_params)
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
        Return the score of the delegate estimator.

        :param X: data frame of the features, shape = (n_samples, n_features)
        :param y: series of the true targets, shape = (n_samples) or (n_samples, \
        n_outputs)
        :param sample_weight:  array-like, sample weights, shape = (n_sample)
        :return: the score of the model
        """
        self._check_parameter_types(X, None)
        return self.delegate_estimator.score(X, y, sample_weight)

    # noinspection PyPep8Naming
    def _prediction_to_series_or_frame(
        self, X: pd.DataFrame, y: MatrixLike[Any]
    ) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            # if we already have a series or data frame, check it and return it
            # unchanged
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


class RegressorWrapperDF(
    RegressorDF[T_Regressor],
    BasePredictorWrapperDF[T_Regressor],
    Generic[T_Regressor],
    ABC,
):
    """
    Wrapper around sklearn regressors that preserves data frames.
    """


class ClassifierWrapperDF(
    ClassifierDF[T_Classifier],
    BasePredictorWrapperDF[T_Classifier],
    Generic[T_Classifier],
    ABC,
):
    """
    Wrapper around sklearn classifiers that preserves data frames.
    """

    @property
    def classes(self) -> Optional[ListLike[Any]]:
        """
        Classes of this classifier after fitting.

        ``None`` if the delegate estimator has no `classes_` property.
        """
        if self.is_fitted:
            return getattr(self.delegate_estimator, "classes_", None)
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
            X, self.delegate_estimator.predict_proba(X)
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
            X, self.delegate_estimator.predict_log_proba(X)
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
            X, self.delegate_estimator.decision_function(X)
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


#
# decorator for easier wrapping of scikit-learn estimators
#


def df_estimator(
    delegate_estimator: Type[T_Estimator] = None,
    *,
    df_estimator_type: Type[
        BaseEstimatorWrapperDF[T_Estimator]
    ] = BaseEstimatorWrapperDF[T_Estimator],
) -> Union[
    Callable[[Type[T_Estimator]], Type[BaseEstimatorWrapperDF[T_Estimator]]],
    Type[BaseEstimatorWrapperDF[T_Estimator]],
]:
    """
    Class decorator wrapping a :class:`sklearn.base.BaseEstimator` in a
    :class:`BaseEstimatorWrapperDF`.
    :param delegate_estimator: the estimator class to wrap
    :param df_estimator_type: optional parameter indicating the \
                              :class:`BaseEstimatorWrapperDF` class to be used for \
                              wrapping; defaults to :class:`BaseEstimatorWrapperDF`
    :return: the resulting `BaseEstimatorWrapperDF` with ``delegate_estimator`` as \
             the delegate estimator
    """

    def _decorate(
        decoratee: Type[T_Estimator]
    ) -> Type[BaseEstimatorWrapperDF[T_Estimator]]:

        # determine the sklearn estimator we are wrapping

        sklearn_base_estimators = [
            base for base in decoratee.__bases__ if issubclass(base, BaseEstimator)
        ]

        if len(sklearn_base_estimators) != 1:
            raise TypeError(
                f"class {decoratee.__name__} must have exactly one base class "
                f"that implements class {BaseEstimator.__name__}"
            )

        sklearn_base_estimator = sklearn_base_estimators[0]

        # wrap the delegate estimator

        @wraps(decoratee, updated=())
        class _DataFrameEstimator(df_estimator_type):
            @classmethod
            def _make_delegate_estimator(cls, *args, **kwargs) -> T_Estimator:
                # noinspection PyArgumentList
                return sklearn_base_estimator(**kwargs)

        return _DataFrameEstimator

    if not issubclass(df_estimator_type, BaseEstimatorWrapperDF):
        raise ValueError(
            f"arg df_transformer_type not a "
            f"{BaseEstimatorWrapperDF.__name__} class: {df_estimator_type}"
        )
    if delegate_estimator is None:
        return _decorate
    else:
        return _decorate(delegate_estimator)
