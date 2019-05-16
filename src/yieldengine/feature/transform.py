import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from yieldengine import Sample

log = logging.getLogger(__name__)


class DataFrameTransformer(ABC, TransformerMixin):
    __slots__ = ["_original_columns"]

    def __init__(self) -> None:
        self._original_columns = None

    @property
    @abstractmethod
    def base_transformer(self) -> TransformerMixin:
        pass

    def is_fitted(self) -> bool:
        return self._original_columns is not None

    @property
    @abstractmethod
    def columns(self) -> pd.Index:
        """
        :returns column labels for arrays returned by the fitted transformer
        """
        pass

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Union[pd.Series, None] = None, **fit_params
    ) -> None:
        self._check_parameter_types(X, y)

        self._original_columns = X.columnms

        # noinspection PyUnresolvedReferences
        self.base_transformer.fit(X, y, **fit_params)

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_parameter_types(X, None)

        if not self.is_fitted():
            raise RuntimeError("attempt to call transform() before fitting")

        # noinspection PyUnresolvedReferences
        transformed = self.base_transformer.transform(X)
        return pd.DataFrame(data=transformed, index=X.index, columns=self.columns)

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.Series, None] = None, **fit_params
    ) -> pd.DataFrame:
        self._check_parameter_types(X, y)

        self._original_columns = X.columns

        transformed = self.base_transformer.fit_transform(X, y, **fit_params)

        return pd.DataFrame(data=transformed, index=X.index, columns=self.columns)

    def fit_transform_sample(self, sample: Sample, **fit_params) -> Sample:
        result = self.fit_transform(X=sample.features, y=sample.target, **fit_params)
        return Sample(
            observations=result.join(sample.target),
            target_name=sample.target_name,
            feature_names=result.columns,
        )

    # noinspection PyPep8Naming
    @staticmethod
    def _check_parameter_types(X: pd.DataFrame, y: Union[pd.Series, None]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be a Series")


class SimpleImputerDF(DataFrameTransformer):
    __slots__ = ["_imputer"]

    def __init__(self, imputer: SimpleImputer) -> None:
        super().__init__()
        self._imputer = imputer

    @property
    def base_transformer(self) -> SimpleImputer:
        return self._imputer

    @property
    def columns(self) -> pd.Index:
        return self._original_columns.delete(
            np.argwhere(np.isnan(self.base_transformer.statistics_))
        )


class OneHotEncoderDF(DataFrameTransformer):
    """
    A one-hot encoder that returns a DataFrame with correct row and column indices
    :param: encoder the underlying sklearn OneHotEncoder
    :param: sep     the feature/value seperator to use when generating column labels
                    for encoded features
    """

    __slots__ = ["_encoder", "_sep"]

    def __init__(self, encoder: OneHotEncoder, sep: str = "=") -> None:
        super().__init__()
        if encoder.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )
        self._encoder = encoder
        self._sep = sep

    @property
    def base_transformer(self) -> OneHotEncoder:
        return self._encoder

    @property
    def columns(self) -> pd.Index:
        return pd.Index(
            [
                f"{feature}={category}"
                for feature, categories in zip(
                    self._original_columns, self.base_transformer.categories_
                )
                for category in categories
            ]
        )
