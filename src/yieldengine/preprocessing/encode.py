"""
Wrap scikit-learn encoders to output dataframes.

The classes defined in this module all inherit from
:class:`yieldengine.df.DataFrameTransformerDF` hence they have ``fit``, ``transform``
and ``fit_transform`` methods and have dataframes as input and output.

:class:`OneHotEncoderDF` wraps :class:`sklearn.preprocessing.OneHotEncoder`.

:class:`OrdinalEncoderDF` wraps :class:`sklearn.preprocessing.OrdinalEncoder`.

:class:`LabelEncoderDF` wraps :class:`sklearn.preprocessing.LabelEncoder`.

:class:`LabelBinarizerDF` wraps :class:`sklearn.preprocessing.LabelBinarizer`.
"""

import logging

import pandas as pd
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
)

from yieldengine.df.transform import DataFrameTransformerWrapper, df_transformer

log = logging.getLogger(__name__)

__all__ = ["OneHotEncoderDF", "OrdinalEncoderDF", "LabelEncoderDF", "LabelBinarizerDF"]


class OneHotEncoderDF(DataFrameTransformerWrapper[OneHotEncoder]):
    """
    One-hot encoder with dataframes as input and output.

    Wrap around :class:`sklearn.preprocessing.OneHotEncoder`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`sklearn.preprocessing.OneHotEncoder`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.base_transformer.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> OneHotEncoder:
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
    Ordinal encoder with dataframes as input and output.

    Wrap around :class:`sklearn.preprocessing.OrdinalEncoder`.

    The parameters are the same as the one passed to
    :class:`sklearn.preprocessing.OrdinalEncoder`.
    """

    pass


@df_transformer
class LabelEncoderDF(LabelEncoder):
    """
    Encode labels with integer values with dataframes as input and output.

    Wrap around :class:`sklearn.preprocessing.LabelEncoder`.

    The parameters are the same as the one passed to
    :class:`sklearn.preprocessing.LabelEncoder`.
    """

    pass


@df_transformer
class LabelBinarizerDF(LabelBinarizer):
    """
    Binarize labels in a one-vs-all fashion with dataframes as input and output.

    Wrap around :class:`sklearn.preprocessing.LabelBinarizer`.

    The parameters are the same as the one passed to
    :class:`sklearn.preprocessing.LabelBinarizer`.
    """

    pass
