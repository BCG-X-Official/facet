# coding=utf-8

import logging
from abc import ABCMeta, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from yieldengine import Sample
from yieldengine.df import DataFrameEstimator

log = logging.getLogger(__name__)

_BaseTransformer = TypeVar(
    "_BaseTransformer", bound=Union[BaseEstimator, TransformerMixin]
)


class DataFrameTransformer(
    DataFrameEstimator[_BaseTransformer], TransformerMixin, metaclass=ABCMeta
):
    """
    Wraps around an sklearn transformer and ensures that the X and y objects passed
    and returned are pandas data frames with valid column names

    :param base_transformer the sklearn transformer to be wrapped
    """

    F_COLUMN_ORIGINAL = "column_original"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._columns_out = None
        self._columns_original = None

    @property
    def base_transformer(self) -> _BaseTransformer:
        return self.base_estimator

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_parameter_types(X, None)

        transformed = self._base_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        self._check_parameter_types(X, y)

        transformed = self._base_fit_transform(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_parameter_types(X, None)

        transformed = self._base_inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_in
        )

    def fit_transform_sample(self, sample: Sample) -> Sample:
        return Sample(
            observations=pd.concat(
                objs=[self.fit_transform(sample.features), sample.target], axis=1
            ),
            target_name=sample.target_name,
        )

    @property
    def columns_original(self) -> pd.Series:
        self._ensure_fitted()
        if self._columns_original is None:
            self._columns_original = (
                self._get_columns_original()
                .rename(self.F_COLUMN_ORIGINAL)
                .rename_axis(index=self.F_COLUMN)
            )
        return self._columns_original

    @property
    def columns_out(self) -> pd.Index:
        return self.columns_original.index

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> _BaseTransformer:
        return cls._make_base_transformer(**kwargs)

    @classmethod
    @abstractmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass

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

    def _transformed_to_df(
        self,
        transformed: Union[pd.DataFrame, np.ndarray],
        index: pd.Index,
        columns: pd.Index,
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


class NDArrayTransformerDF(
    DataFrameTransformer[_BaseTransformer], Generic[_BaseTransformer], metaclass=ABCMeta
):
    """
    Special case of DataFrameTransformer where the base transformer does not accept
    data frames, but only numpy ndarrays
    """

    # noinspection PyPep8Naming
    def _base_fit(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params) -> None:
        # noinspection PyUnresolvedReferences
        self.base_transformer.fit(X.values, y.values, **fit_params)

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
    DataFrameTransformer[_BaseTransformer], Generic[_BaseTransformer], metaclass=ABCMeta
):
    """
    All output columns of a ColumnPreservingTransformer have the same names as their
    associated input columns
    """

    @abstractmethod
    def _get_columns_out(self) -> pd.Index:
        """
        :returns column labels for arrays returned by the fitted transformer
        """
        pass

    def _get_columns_original(self) -> pd.Series:
        columns_out = self._get_columns_out()
        return pd.Series(index=columns_out, data=columns_out.values)


class ConstantColumnTransformer(
    ColumnPreservingTransformer[_BaseTransformer],
    Generic[_BaseTransformer],
    metaclass=ABCMeta,
):
    """
    A ConstantColumnTransformer does not add, remove, or rename any of the input columns
    """

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in


# Decorator to easily create ConstantColumnTransformers, example:
# see: src/tests/transform/test_constant_column_transformer
def constant_column_transformer(source_transformer: type) -> Callable:
    def decorate(class_in: type) -> type:
        def _make_base_transformer(**kwargs) -> source_transformer:
            return source_transformer(**kwargs)

        def __init__(self, **kwargs) -> None:
            ConstantColumnTransformer[source_transformer].__init__(self, **kwargs)

        class_in.__init__ = __init__
        class_in._make_base_transformer = _make_base_transformer
        return class_in

    return decorate
