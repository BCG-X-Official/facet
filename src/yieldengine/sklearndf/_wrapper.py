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
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)

from yieldengine import ListLike, MatrixLike
from yieldengine.sklearndf import (
    DataFrameClassifier,
    DataFrameEstimator,
    DataFramePredictor,
    DataFrameRegressor,
    DataFrameTransformer,
)

log = logging.getLogger(__name__)

#
# type variables
#

T_BaseEstimator = TypeVar("T_BaseEstimator", bound=BaseEstimator)
T_BaseTransformer = TypeVar(
    "T_BaseTransformer", bound=Union[BaseEstimator, TransformerMixin]
)
T_BasePredictor = TypeVar(
    "T_BasePredictor", bound=Union[RegressorMixin, ClassifierMixin]
)
T_BaseRegressor = TypeVar("T_BaseRegressor", bound=RegressorMixin)
T_BaseClassifier = TypeVar("T_BaseClassifier", bound=ClassifierMixin)


#
# base wrapper classes
#


class DataFrameEstimatorWrapper(
    DataFrameEstimator, BaseEstimator, Generic[T_BaseEstimator]
):
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
    def _make_base_estimator(cls, **kwargs) -> T_BaseEstimator:
        pass

    @property
    def base_estimator(self) -> T_BaseEstimator:
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

    def set_params(self, **kwargs) -> "DataFrameEstimatorWrapper[T_BaseEstimator]":
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
    ) -> "DataFrameEstimatorWrapper[T_BaseEstimator]":
        """
        Fit the base estimator.

        :param X: data frame to fit the estimator
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
    ) -> T_BaseEstimator:
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


class DataFrameTransformerWrapper(
    DataFrameTransformer, DataFrameEstimatorWrapper[T_BaseTransformer], ABC
):
    """
    Wraps a :class:`sklearn.base.TransformerMixin` and ensures that the X and y
    objects passed and returned are pandas data frames with valid column names.

    Implementations must define ``_make_base_estimator`` and
    ``_get_columns_original``.

    :param `**kwargs`: parameters of scikit-learn transformer to be wrapped
    """

    F_COLUMN_OUT = "column_out"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._columns_out = None
        self._columns_original = None

    @property
    def base_transformer(self) -> T_BaseTransformer:
        """The base scikit-learn transformer"""
        return self.base_estimator

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Call the transform method of the base transformer ``self.base_transformer``.

        :param X: data frame to transform
        :return: transformed data frame
        """
        self._check_parameter_types(X, None)

        transformed = self._base_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        """Call the ``fit_transform`` method of ``self.base_transformer``.

        :param X: data frame to transform
        :param y: series of training targets
        :param fit_params: parameters passed to the fit method of the base transformer
        :return: data frame of transformed sample
        """
        self._check_parameter_types(X, y)

        transformed = self._base_fit_transform(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformations in reverse order on the base transformer.

        All estimators in the pipeline must support ``inverse_transform``.
        :param X: data frame of samples
        :return: data frame of inverse-transformed samples
        """

        self._check_parameter_types(X, None)

        transformed = self._base_inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_in
        )

    @property
    def columns_original(self) -> pd.Series:
        """Series mapping output column names to the original columns names.

        Series with index the name of the output columns and with values the
        original name of the column."""
        self._ensure_fitted()
        if self._columns_original is None:
            self._columns_original = (
                self._get_columns_original()
                .rename(self.F_COLUMN_IN)
                .rename_axis(index=self.F_COLUMN_OUT)
            )
        return self._columns_original

    @abstractmethod
    def _get_columns_original(self) -> pd.Series:
        """
        :return: a mapping from this transformer's output columns to the original
        columns as a series
        """
        pass

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> None:
        super()._post_fit(X=X, y=y, **fit_params)
        self._columns_out = None
        self._columns_original = None

    @staticmethod
    def _transformed_to_df(
        transformed: Union[pd.DataFrame, np.ndarray], index: pd.Index, columns: pd.Index
    ):
        if isinstance(transformed, pd.DataFrame):
            return transformed
        else:
            return pd.DataFrame(data=transformed, index=index, columns=columns)

    # noinspection PyPep8Naming
    def _base_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.base_transformer.transform(X)

    # noinspection PyPep8Naming
    def _base_fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.base_transformer.fit_transform(X, y, **fit_params)

    # noinspection PyPep8Naming
    def _base_inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.base_transformer.inverse_transform(X)


class DataFramePredictorWrapper(
    DataFramePredictor, DataFrameEstimatorWrapper[T_BasePredictor], ABC
):
    """
    Base class for sklearn regressors and classifiers that preserve data frames

    :param `**kwargs`: arguments passed to :class:`.DataFrameEstimator` in ``__init__``
    """

    F_PREDICTION = "prediction"

    @property
    def n_outputs(self) -> int:
        """
        Number of outputs predicted by this predictor.

        Defaults to 1 if base predictor does not define property ``n_outputs_``.
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
        Compute the prediction as a series or a data frame.

        For single-output problems, return a series, fro multi-output problems,
        return a data frame.

        :param X: the data frame of features
        :param predict_params: additional arguments passed to the ``predict`` method \
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
        :param fit_params: additional arguments passed to the the ``predict`` method
          of the base estimator
        :return: series of the predictions for X
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


class DataFrameRegressorWrapper(
    DataFrameRegressor, DataFramePredictorWrapper[T_BaseRegressor], ABC
):
    """
    Wrapper around sklearn regressors that preserves data frames.
    """


class DataFrameClassifierWrapper(
    DataFrameClassifier, DataFramePredictorWrapper[T_BaseClassifier], ABC
):
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


#
# specialised transformer wrappers
#


class NDArrayTransformerDF(
    DataFrameTransformerWrapper[T_BaseTransformer], Generic[T_BaseTransformer], ABC
):
    """
    `DataFrameTransformer` whose base transformer only accepts numpy ndarrays.

    Wraps around the base transformer and converts the data frame to an array when
    needed.
    """

    # noinspection PyPep8Naming
    def _base_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> T_BaseTransformer:
        # noinspection PyUnresolvedReferences
        return self.base_transformer.fit(X.values, y.values, **fit_params)

    # noinspection PyPep8Naming
    def _base_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.base_transformer.transform(X.values)

    # noinspection PyPep8Naming
    def _base_fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.base_transformer.fit_transform(X.values, y.values, **fit_params)

    # noinspection PyPep8Naming
    def _base_inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.base_transformer.inverse_transform(X.values)


class ColumnPreservingTransformer(
    DataFrameTransformerWrapper[T_BaseTransformer], Generic[T_BaseTransformer], ABC
):
    """
    Transforms a data frame without changing column names, but possibly removing
    columns.

    All output columns of a :class:`ColumnPreservingTransformer` have the same names as
    their associated input columns. Some columns can be removed.
    Implementations must define ``_make_base_estimator`` and ``_get_columns_out``.
    """

    @abstractmethod
    def _get_columns_out(self) -> pd.Index:
        # return column labels for arrays returned by the fitted transformer.
        pass

    def _get_columns_original(self) -> pd.Series:
        # return the series with output columns in index and output columns as values
        columns_out = self._get_columns_out()
        return pd.Series(index=columns_out, data=columns_out.values)


class ConstantColumnTransformer(
    ColumnPreservingTransformer[T_BaseTransformer], Generic[T_BaseTransformer], ABC
):
    """
    Transforms a data frame keeping exactly the same columns.

    A ConstantColumnTransformer does not add, remove, or rename any of the input
    columns. Implementations must define ``_make_base_estimator``.
    """

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in


#
# decorator for easier wrapping of scikit-learn estimators
#


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
    Class decorator wrapping a :class:`sklearn.base.BaseEstimator` in a
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

        # wrap the base estimator

        @wraps(decoratee, updated=())
        class _DataFrameEstimator(df_estimator_type):
            @classmethod
            def _make_base_estimator(cls, **kwargs) -> T_BaseEstimator:
                # noinspection PyArgumentList
                return sklearn_base_estimator(**kwargs)

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
