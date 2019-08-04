#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Wrap scikit-learn `BaseEstimator` to return dataframes instead of numpy arrays.

The abstract class :class:`BaseEstimatorDF` wraps
:class:`~sklearn.base.BaseEstimator` so that the ``predict``
and ``transform`` methods of the implementations return dataframe.
:class:`BaseEstimatorDF` has an attribute :attr:`~BaseEstimatorDF.columns_in`
which is the index of the columns of the input dataframe.
"""

import logging
from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    clone,
    RegressorMixin,
    TransformerMixin,
)

from gamma import ListLike, Sample

log = logging.getLogger(__name__)

__all__ = [
    "BaseEstimatorDF",
    "BasePredictorDF",
    "ClassifierDF",
    "RegressorDF",
    "TransformerDF",
]

#
# type variables
#

# noinspection PyShadowingBuiltins
_T = TypeVar("_T")

T_Estimator = TypeVar("T_Estimator", bound=BaseEstimator)
T_Transformer = TypeVar("T_Transformer", bound=TransformerMixin)
T_Predictor = TypeVar("T_Predictor", bound=Union[RegressorMixin, ClassifierMixin])
T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)
T_Classifier = TypeVar("T_Classifier", bound=ClassifierMixin)

#
# class definitions
#


class BaseEstimatorDF(ABC, Generic[T_Estimator]):
    """
    #todo find a good description

    Implementations must define a ``fit`` method and an ``is_fitted`` property.
    """

    F_COLUMN_IN = "column_in"

    def __init__(self) -> None:
        super().__init__()
        if not isinstance(self, BaseEstimator):
            raise TypeError(
                f"class {type(self).__name__} is required to inherit from class "
                f"{BaseEstimator.__name__}"
            )
        self._columns_in = None

    @property
    def delegate_estimator(self) -> T_Estimator:
        """
        If this estimator is derived from a non-data frame estimator, return the
        original estimator; otherwise, return ``self``.

        :return: the original estimator that this estimator delegates to
        """
        return cast(T_Estimator, self)

    # noinspection PyPep8Naming
    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "BaseEstimatorDF":
        pass

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """`True` if the delegate estimator is fitted, else `False`"""
        pass

    @property
    def columns_in(self) -> pd.Index:
        """The names of the input columns this estimator was fitted on"""
        self._ensure_fitted()
        return self._get_columns_in().rename(self.F_COLUMN_IN)

    def clone(self: _T) -> _T:
        """
        Make an unfitted clone of this estimator.
        :return: the unfitted clone
        """
        return clone(self)

    def _ensure_fitted(self) -> None:
        # raise an AttributeError if this transformer is not fitted
        if not self.is_fitted:
            raise AttributeError("estimator is not fitted")

    @abstractmethod
    def _get_columns_in(self) -> pd.Index:
        # get the input columns as a pandas Index
        pass


class BasePredictorDF(BaseEstimatorDF[T_Predictor], Generic[T_Predictor], ABC):
    @property
    @abstractmethod
    def n_outputs(self) -> int:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        pass


class TransformerDF(
    BaseEstimatorDF[T_Transformer], TransformerMixin, Generic[T_Transformer], ABC
):
    F_COLUMN_OUT = "column_out"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._columns_original = None

    # noinspection PyPep8Naming
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        return self.fit(X, y, **fit_params).transform(X)

    # noinspection PyPep8Naming
    @abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform_sample(self, sample: Sample) -> Sample:
        """
        Fit and transform with input/output as a :class:`~yieldengine.Sample` object.

        :param sample: sample used as input
        :return: transformed sample
        """
        return Sample(
            observations=pd.concat(
                objs=[self.fit_transform(sample.features), sample.target], axis=1
            ),
            target_name=sample.target_name,
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

    @property
    def columns_out(self) -> pd.Index:
        """The `pd.Index` of names of the output columns."""
        self._ensure_fitted()
        return self._get_columns_out().rename(self.F_COLUMN_OUT)

    @abstractmethod
    def _get_columns_original(self) -> pd.Series:
        """
        :return: a mapping from this transformer's output columns to the original
        columns as a series
        """
        pass

    def _get_columns_out(self) -> pd.Index:
        # default behaviour: get index returned by columns_original
        return self.columns_original.index


class RegressorDF(
    BasePredictorDF[T_Regressor], RegressorMixin, Generic[T_Regressor], ABC
):
    """
    Sklearn regressor that preserves data frames.
    """


class ClassifierDF(
    BasePredictorDF[T_Classifier], ClassifierMixin, Generic[T_Classifier], ABC
):
    """
    Sklearn classifier that preserves data frames.
    """

    @property
    @abstractmethod
    def classes(self) -> Optional[ListLike[Any]]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        pass
