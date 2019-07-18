"""Wrap sklearn encoders to output dataframes."""

import logging

import pandas as pd
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)

from yieldengine.df.transform import DataFrameTransformer, df_transformer

log = logging.getLogger(__name__)

__all__ = ["OneHotEncoderDF", "OrdinalEncoderDF", "LabelEncoderDF", "LabelBinarizerDF"]


class OneHotEncoderDF(DataFrameTransformer[OneHotEncoder]):
    """
    One-hot encoder that returns a DataFrame.

    The parameters are the same as the one passed to sklearn `OneHotEncoder`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.base_transformer.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> OneHotEncoder:
        return OneHotEncoder(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        """
        Return the series mapping output column names to original columns names.

        :return: the series with index the column names of the output dataframe and
        values the corresponding input column names.
        """
        return pd.Series(
            index=pd.Index(self.base_transformer.get_feature_names(self.columns_in)),
            data=[
                column_original
                for column_original, category in zip(
                    self.columns_in, self.base_transformer.categories_
                )
                for _ in category
            ],
        )


@df_transformer
class OrdinalEncoderDF(OrdinalEncoder):
    """
    Ordinal encoder that returns a DataFrame.

    The parameters are the same as the one passed to sklearn `OrdinalEncoder`.
    """

    pass


@df_transformer
class LabelEncoderDF(LabelEncoder):
    """
    Encode labels with integer values and return a DataFrame.

    The parameters are the same as the one passed to sklearn `LabelEncoder`.
    """

    pass


@df_transformer
class LabelBinarizerDF(LabelBinarizer):
    """
    Binarize labels in a one-vs-all fashion and return a DataFrame.

    The parameters are the same as the one passed to sklearn `LabelBinarizer`.
    """

    pass
