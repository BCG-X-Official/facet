"""Wrapper around sklearn `ColumnTransformer` compatible with dataframes."""

import logging
from functools import reduce
from typing import *

import pandas as pd
from sklearn.compose import ColumnTransformer

from yieldengine.df.transform import DataFrameTransformer

log = logging.getLogger(__name__)

__all__ = ["ColumnTransformerDF"]


class ColumnTransformerDF(DataFrameTransformer[ColumnTransformer]):
    """
    Wrap sklearn ```ColumnTransformer``` and return a DataFrame.

    Like a sklearn `ColumnTransformer`, self has a `transformers` parameter (None by
    default) which is a list of tuple of the form (name, transformer, column(s)),
    but here all the transformers must be of type
    :class:`~yieldengine.df.transform.DataFrameTransformer`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # noinspection PyTypeChecker
        column_transformer = self.base_transformer

        if column_transformer.remainder != "drop":
            raise ValueError(
                f"arg column_transformer with unsupported remainder attribute "
                f"({column_transformer.remainder})"
            )

        if not (
            all(
                [
                    isinstance(transformer, DataFrameTransformer)
                    for _, transformer, _ in column_transformer.transformers
                ]
            )
        ):
            raise ValueError(
                "arg column_transformer must only contain instances of "
                "DataFrameTransformer"
            )

        self._columnTransformer = column_transformer

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> ColumnTransformer:
        return ColumnTransformer(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        return reduce(
            lambda x, y: x.append(y),
            (
                df_transformer.columns_original
                for df_transformer in self._inner_transformers()
            ),
        )

    def _inner_transformers(self) -> Iterable[DataFrameTransformer]:
        return (
            df_transformer
            for _, df_transformer, columns in self.base_transformer.transformers_
            if len(columns) > 0
            if df_transformer != "drop"
        )
