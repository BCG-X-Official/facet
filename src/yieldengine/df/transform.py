# coding=utf-8
"""Base classes with wrapper around sklearn transformers."""

import logging
from abc import ABCMeta, abstractmethod
from types import new_class
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
    and returned are pandas data frames with valid column names.

    Implementations must define `_make_base_transformer` and `_get_columns_original`.

    :param: base_transformer the sklearn transformer to be wrapped
    """

    F_COLUMN_ORIGINAL = "column_original"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._columns_out = None
        self._columns_original = None

    @property
    def base_transformer(self) -> _BaseTransformer:
        """The base sklean transformer"""
        return self.base_estimator

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calls the transform method of the base transformer `self.base_transformer`.

        :param X: dataframe to transform
        :return: transformed dataframe
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
        """Calls the fit_transform method of the base transformer
        `self.base_transformer`.

        :param X: dataframe to transform
        :param y: series of training targets
        :param fit_params: parameters passed to the fit method of the base transformer
        :return: dataframe of transformed sample
        """
        self._check_parameter_types(X, y)

        transformed = self._base_fit_transform(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_out
        )

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformations in reverse order on the base tranformer.

        All estimators in the pipeline must support ``inverse_transform``.
        :param X: dataframe of samples
        :return: dataframe of inversed samples
        """

        self._check_parameter_types(X, None)

        transformed = self._base_inverse_transform(X)

        return self._transformed_to_df(
            transformed=transformed, index=X.index, columns=self.columns_in
        )

    def fit_transform_sample(self, sample: Sample) -> Sample:
        """
        Fit and transform with input and output being a `Sample` object.
        :param sample: `Sample` object used as input
        :return: transformed `Sample` object
        """
        return Sample(
            observations=pd.concat(
                objs=[self.fit_transform(sample.features), sample.target], axis=1
            ),
            target_name=sample.target_name,
        )

    @property
    def columns_original(self) -> pd.Series:
        """Series with index the name of the output columns and with values the
        original name of the column"""
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
        """The `pd.Index` of name of the output columns"""
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
    """Abstract base class for a `DataFrameTransformer`.

    All output columns of a ColumnPreservingTransformer have the same names as their
    associated input columns. It could be however that some columns are removed.
    Implementations must define `_make_base_transformer` and `_get_columns_out`.
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
    """Abstract base class for a `DataFrameTransformer`.

    A ConstantColumnTransformer does not add, remove, or rename any of the input
    columns. Implementations must define `_make_base_transformer`.
    """

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in


def constant_column_transformer(
    cls: Type[_BaseTransformer]
) -> Type[ConstantColumnTransformer[_BaseTransformer]]:
    def _init_class_namespace(namespace: Dict[str, Any]) -> None:
        # noinspection PyProtectedMember
        namespace[
            ConstantColumnTransformer._make_base_transformer.__name__
        ] = lambda **kwargs: cls(**kwargs)

    return cast(
        Type[ConstantColumnTransformer[_BaseTransformer]],
        new_class(
            name=cls.__name__,
            bases=(ConstantColumnTransformer,),
            exec_body=_init_class_namespace,
        ),
    )
