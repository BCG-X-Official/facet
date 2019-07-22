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
from typing import Any, List, Optional, Union

import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
)

from gamma import ListLike, Sample

log = logging.getLogger(__name__)


class DataFrameEstimator(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        if not isinstance(self, BaseEstimator):
            raise TypeError(
                f"class {type(self).__name__} is required to inherit from class "
                f"{BaseEstimator.__name__}"
            )
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


class DataFramePredictor(DataFrameEstimator, ABC):
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


class _DataFramePredictor(DataFramePredictor):
    """Dummy data frame predictor class, for type hinting only."""

    @property
    def n_outputs(self) -> int:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        """Dummy implementation."""
        raise NotImplementedError()

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "DataFramePredictor":
        """Dummy implementation."""
        raise NotImplementedError()

    @property
    def is_fitted(self) -> bool:
        """Dummy implementation."""
        raise NotImplementedError()

    @property
    def columns_in(self) -> pd.Index:
        """Dummy implementation."""
        raise NotImplementedError()


class DataFrameTransformer(DataFrameEstimator, TransformerMixin, ABC):
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        return self.fit(X, y, **fit_params).transform(X)

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


class DataFrameRegressor(DataFramePredictor, RegressorMixin):
    """
    Sklearn regressor that preserves data frames.
    """


class DataFrameClassifier(DataFramePredictor, ClassifierMixin):
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
