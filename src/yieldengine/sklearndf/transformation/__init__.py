"""
Preprocessing
"""

import logging
from functools import reduce
from typing import Iterable, Type

import numpy as np
import pandas as pd
from sklearn import compose, impute, preprocessing
from sklearn.base import BaseEstimator

from yieldengine.sklearndf._wrapper import (
    ColumnPreservingTransformer,
    ConstantColumnTransformer,
    DataFrameTransformerWrapper,
    df_estimator,
)

log = logging.getLogger(__name__)


#
# decorator for wrapping the sklearn classifier classes
#


def _df_transformer(base_transformer: Type[BaseEstimator]):
    return df_estimator(
        base_estimator=base_transformer, df_estimator_type=ConstantColumnTransformer
    )


#
# preprocessing
#


@_df_transformer
class MaxAbsScalerDF(preprocessing.MaxAbsScaler):
    pass


@_df_transformer
class MinMaxScalerDF(preprocessing.MinMaxScaler):
    pass


@_df_transformer
class NormalizerDF(preprocessing.Normalizer):
    pass


@_df_transformer
class PowerTransformerDF(preprocessing.PowerTransformer):
    pass


@_df_transformer
class QuantileTransformerDF(preprocessing.QuantileTransformer):
    pass


@_df_transformer
class RobustScalerDF(preprocessing.RobustScaler):
    pass


@_df_transformer
class StandardScalerDF(preprocessing.StandardScaler):
    pass


@_df_transformer
class KernelCentererDF(preprocessing.KernelCenterer):
    pass


@_df_transformer
class FunctionTransformerDF(preprocessing.FunctionTransformer):
    pass


class OneHotEncoderDF(DataFrameTransformerWrapper[preprocessing.OneHotEncoder]):
    """
    One-hot encoder with dataframes as input and output.

    Wrap around :class:`preprocessing.OneHotEncoder`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`preprocessing.OneHotEncoder`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.base_transformer.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> preprocessing.OneHotEncoder:
        return preprocessing.OneHotEncoder(**kwargs)

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


@_df_transformer
class OrdinalEncoderDF(preprocessing.OrdinalEncoder):
    """
    Ordinal encoder with dataframes as input and output.

    Wrap around :class:`preprocessing.OrdinalEncoder`.

    The parameters are the same as the one passed to
    :class:`preprocessing.OrdinalEncoder`.
    """

    pass


@_df_transformer
class LabelEncoderDF(preprocessing.LabelEncoder):
    """
    Encode labels with integer values with dataframes as input and output.

    Wrap around :class:`preprocessing.LabelEncoder`.

    The parameters are the same as the one passed to
    :class:`preprocessing.LabelEncoder`.
    """

    pass


@_df_transformer
class LabelBinarizerDF(preprocessing.LabelBinarizer):
    """
    Binarize labels in a one-vs-all fashion with dataframes as input and output.

    Wrap around :class:`preprocessing.LabelBinarizer`.

    The parameters are the same as the one passed to
    :class:`preprocessing.LabelBinarizer`.
    """

    pass


class PolynomialFeaturesDF(
    ColumnPreservingTransformer[preprocessing.PolynomialFeatures]
):
    def _get_columns_out(self) -> pd.Index:
        return pd.Index(
            data=self.base_transformer.get_feature_names(input_features=self.columns_in)
        )

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> preprocessing.PolynomialFeatures:
        return preprocessing.PolynomialFeatures(**kwargs)


#
# compose
#


class ColumnTransformerDF(DataFrameTransformerWrapper[compose.ColumnTransformer]):
    """
    Wrap :class:`sklearn.compose.ColumnTransformer` and return a DataFrame.

    Like :class:`~sklearn.compose.ColumnTransformer`, it has a ``transformers``
    parameter
    (``None`` by default) which is a list of tuple of the form (name, transformer,
    column(s)),
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
                    isinstance(transformer, DataFrameTransformerWrapper)
                    for _, transformer, _ in column_transformer.transformers
                ]
            )
        ):
            raise ValueError(
                "arg column_transformer must only contain instances of "
                "DataFrameTransformerWrapper"
            )

        self._columnTransformer = column_transformer

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> compose.ColumnTransformer:
        return compose.ColumnTransformer(**kwargs)

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

    def _inner_transformers(self) -> Iterable[DataFrameTransformerWrapper]:
        return (
            df_transformer
            for _, df_transformer, columns in self.base_transformer.transformers_
            if len(columns) > 0
            if df_transformer != "drop"
        )


#
# impute
#


class SimpleImputerDF(ColumnPreservingTransformer[impute.SimpleImputer]):
    """
    Impute missing values with dataframes as input and output.

    Wrap around :class:`impute.SimpleImputer`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`impute.SimpleImputer`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> impute.SimpleImputer:
        return impute.SimpleImputer(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        stats = self.base_transformer.statistics_
        if issubclass(stats.dtype.type, float):
            nan_mask = np.isnan(stats)
        else:
            nan_mask = [
                x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
            ]
        return self.columns_in.delete(np.argwhere(nan_mask))


class MissingIndicatorDF(DataFrameTransformerWrapper[impute.MissingIndicator]):
    """
    Indicate missing values with dataframes as input and output.

    Wrap :class:`impute.MissingIndicatorDF`. The ``fit``,
    ``transform`` and ``fit_transform`` methods accept and return dataframes.
    The parameters are the same as the one passed to
    :class:`impute.MissingIndicator`.

    The parameters are the same as the one passed to
    :class:`impute.MissingIndicator`.
    """

    def __init__(
        self,
        missing_values=np.nan,
        features="missing-only",
        sparse="auto",
        error_on_new=True,
        **kwargs,
    ) -> None:
        super().__init__(
            missing_values=missing_values,
            features=features,
            sparse=sparse,
            error_on_new=error_on_new,
            **kwargs,
        )

    @classmethod
    def _make_base_estimator(cls, **kwargs) -> impute.MissingIndicator:
        return impute.MissingIndicator(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        columns_original: np.ndarray = self.columns_in[
            self.base_transformer.features_
        ].values
        columns_out = pd.Index([f"{name}__missing" for name in columns_original])
        return pd.Series(index=columns_out, data=columns_original)
