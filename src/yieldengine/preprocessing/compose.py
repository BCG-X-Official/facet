import logging
from functools import reduce
from typing import *

import pandas as pd
from sklearn.compose import ColumnTransformer

from yieldengine.df.transform import DataFrameTransformer

log = logging.getLogger(__name__)

__all__ = ["ColumnTransformerDF"]


class ColumnTransformerDF(DataFrameTransformer[ColumnTransformer]):
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
        :return: a list of the original features from which the output of this
        transformer is derived. The list is aligned with the columns returned by
        method :meth" `output columns` and this has the same length
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
