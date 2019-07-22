# coding=utf-8
"""
Wrappers around scikit-learn transformers.

Create classes analogue to :class:`sklearn.base.TransformerMixin`. The outcome are
classes the have ``fit``, ``transform`` and ``fit_transform`` methods that accept and
return dataframes (while scikit-learn transformers alwsays return a numpy array and
sometimes do not accept a dataframe as input).

:class:`DataFrameTransformer` is an ABC class that wraps sklearn
:class:`sklearn.base.TransformerMixin`.

The attribute :attr:`DataFrameTransformer.columns_original` is a
:class:`pandas.Series` with index the output columns and values the input column names.
"""

import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from yieldengine import Sample
from yieldengine.df import DataFrameEstimator, DataFrameEstimatorWrapper, df_estimator

log = logging.getLogger(__name__)

T_BaseEstimator = TypeVar("T_BaseEstimator", bound=BaseEstimator)

T_BaseTransformer = TypeVar(
    "T_BaseTransformer", bound=Union[BaseEstimator, TransformerMixin]
)


class DataFrameTransformer(DataFrameEstimator, TransformerMixin, ABC):
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        pass

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


class DataFrameTransformerWrapper(
    DataFrameTransformer, DataFrameEstimatorWrapper[T_BaseTransformer], ABC
):
    """
    Wraps a :class:`sklearn.base.TransformerMixin` and ensures that the X and y
    objects passed and returned are pandas data frames with valid column names.

    Implementations must define ``_make_base_estmiator`` and
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
        """Call the ``fit_transform`` method of ``self.base_transformer``.

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
        """Apply inverse transformations in reverse order on the base transformer.

        All estimators in the pipeline must support ``inverse_transform``.
        :param X: dataframe of samples
        :return: dataframe of inversed samples
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


class NDArrayTransformerDF(
    DataFrameTransformerWrapper[T_BaseTransformer], Generic[T_BaseTransformer], ABC
):
    """
    `DataFrameTransformer` whose base transformer only accepts numpy ndarrays.

    Wraps around the base transformer and converts the dataframe to an array when
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
    Transforms a dataframe without changing column names but can remove columns.

    All output columns of a :class:`ColumnPreservingTransformer` have the same names as
    their
    associated input columns. Some columns can be removed.
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
    Transforms a dataframe keeping exactly the same columns.

    A ConstantColumnTransformer does not add, remove, or rename any of the input
    columns. Implementations must define ``_make_base_estimator``.
    """

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in


def df_transformer(
    base_transformer: Type[T_BaseTransformer] = None,
    *,
    df_transformer_type: Type[
        DataFrameTransformerWrapper[T_BaseTransformer]
    ] = ConstantColumnTransformer,
) -> Union[
    Callable[
        [Type[T_BaseTransformer]], Type[DataFrameTransformerWrapper[T_BaseTransformer]]
    ],
    Type[DataFrameTransformerWrapper[T_BaseTransformer]],
]:
    """
    Class decorator wrapping a :class:`sklearn.base.TransformerMixin` into a
    :class:`DataFrameTransformer`.

    :param base_transformer: the transformer class to wrap
    :param df_transformer_type: optional parameter indicating the \
                                :class:`DataFrameTransformer` class to be used for
                                wrapping; \
                                defaults to :class:`ConstantColumnTransformer`
    :return: the resulting `DataFrameTransformer` with ``base_transformer`` \
             as the base transformer
    """

    return cast(
        Union[
            Callable[
                [Type[T_BaseTransformer]],
                Type[DataFrameTransformerWrapper[T_BaseTransformer]],
            ],
            Type[DataFrameTransformerWrapper[T_BaseTransformer]],
        ],
        df_estimator(
            base_estimator=base_transformer, df_estimator_type=df_transformer_type
        ),
    )
