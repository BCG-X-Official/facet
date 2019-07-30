# coding=utf-8
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
    "T_Classifier",
    "T_ClassifierDF",
    "T_Estimator",
    "T_Predictor",
    "T_PredictorDF",
    "T_Regressor",
    "T_RegressorDF",
    "T_Transformer",
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
        return cast(BaseEstimator, self)

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
    @abstractmethod
    def columns_in(self) -> pd.Index:
        """The names of the input columns this estimator was fitted on"""
        pass

    def clone(self: _T) -> _T:
        """
        Make an unfitted clone of this estimator.
        :return: the unfitted clone
        """
        return clone(self)


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
    @abstractmethod
    def columns_original(self) -> pd.Series:
        pass

    @property
    def columns_out(self) -> pd.Index:
        """The `pd.Index` of names of the output columns."""
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


#
# type variables for df predictors
#

T_PredictorDF = TypeVar("T_PredictorDF", bound=BasePredictorDF)
T_RegressorDF = TypeVar("T_RegressorDF", bound=RegressorDF)
T_ClassifierDF = TypeVar("T_ClassifierDF", bound=ClassifierDF)
