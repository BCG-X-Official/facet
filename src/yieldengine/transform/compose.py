import logging
from functools import reduce
from itertools import chain
from typing import *

import pandas as pd
from sklearn.compose import ColumnTransformer

from yieldengine.transform import DataFrameTransformer

log = logging.getLogger(__name__)


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
    def _make_base_transformer(cls, **kwargs) -> ColumnTransformer:
        return ColumnTransformer(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        # construct the index from the columns in the fitted transformers
        return pd.Index(
            data=chain(
                *[
                    df_transformer.columns_out
                    for df_transformer in self._inner_transformers()
                ]
            ),
            name=DataFrameTransformer.F_COLUMN,
        )

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
            for _, df_transformer, _ in self.base_transformer.transformers_
            if df_transformer != "drop"
        )
