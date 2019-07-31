"""
Specialised transformer wrappers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional

import numpy as np
import pandas as pd

from gamma.sklearndf import T_Transformer
from gamma.sklearndf._wrapper import TransformerWrapperDF

log = logging.getLogger(__name__)


class NDArrayTransformerWrapperDF(
    TransformerWrapperDF[T_Transformer], Generic[T_Transformer], ABC
):
    """
    `TransformerDF` whose delegate transformer only accepts numpy ndarrays.

    Wraps around the delegate transformer and converts the data frame to an array when
    needed.
    """

    # noinspection PyPep8Naming
    def _fit(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> T_Transformer:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.fit(X.values, y.values, **fit_params)

    # noinspection PyPep8Naming
    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.transform(X.values)

    # noinspection PyPep8Naming
    def _fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params
    ) -> np.ndarray:
        return self.delegate_estimator.fit_transform(X.values, y.values, **fit_params)

    # noinspection PyPep8Naming
    def _inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.delegate_estimator.inverse_transform(X.values)


class ColumnSubsetTransformerWrapperDF(
    TransformerWrapperDF[T_Transformer], Generic[T_Transformer], ABC
):
    """
    Transforms a data frame without changing column names, but possibly removing
    columns.

    All output columns of a :class:`ColumnSubsetTransformerWrapperDF` have the same
    names as their associated input columns. Some columns can be removed.
    Implementations must define ``_make_delegate_estimator`` and ``_get_columns_out``.
    """

    @abstractmethod
    def _get_columns_out(self) -> pd.Index:
        # return column labels for arrays returned by the fitted transformer.
        pass

    def _get_columns_original(self) -> pd.Series:
        # return the series with output columns in index and output columns as values
        columns_out = self._get_columns_out()
        return pd.Series(index=columns_out, data=columns_out.values)


class ColumnPreservingTransformerWrapperDF(
    ColumnSubsetTransformerWrapperDF[T_Transformer], Generic[T_Transformer], ABC
):
    """
    Transforms a data frame keeping exactly the same columns.

    A ``ColumnPreservingTransformerWrapperDF`` does not add, remove, or rename any of the
    input columns.
    """

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in


class BaseDimensionalityReductionWrapperDF(TransformerWrapperDF):
    def _get_columns_original(self) -> pd.Series:
        raise AttributeError(
            "columns_original is not defined for dimensionality reduction transformers"
        )


class AnonymousDimensionalityReductionWrapperDF(BaseDimensionalityReductionWrapperDF):
    def _get_columns_out(self) -> pd.Index:
        # todo: implement this
        pass


class NamedDimensionalityReductionWrapperDF(BaseDimensionalityReductionWrapperDF):
    def _get_columns_out(self) -> pd.Index:
        # todo: implement this
        pass


class FeatureSelectionWrapperDF:
    # todo: implement this
    pass
