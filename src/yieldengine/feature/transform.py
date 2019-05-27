import logging
from abc import ABC, abstractmethod
from itertools import chain
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

log = logging.getLogger(__name__)


_BaseTransformer = TypeVar(
    "_BaseTransformer", bound=Union[BaseEstimator, TransformerMixin]
)


class DataFrameTransformer(
    ABC, BaseEstimator, TransformerMixin, Generic[_BaseTransformer]
):
    """
    Wraps around an sklearn transformer and ensures that the X and y objects passed
    and returned are pandas data frames with valid column names

    :param base_transformer the sklearn transformer to be wrapped
    """

    def __init__(self, **kwargs) -> None:
        super(BaseEstimator).__init__()
        super(TransformerMixin).__init__()
        self._base_transformer = type(self)._make_base_transformer(**kwargs)
        self._original_columns = None

    @classmethod
    @abstractmethod
    def _make_base_transformer(cls, **kwargs) -> _BaseTransformer:
        pass

    @property
    def base_transformer(self) -> _BaseTransformer:
        return self._base_transformer

    def is_fitted(self) -> bool:
        return self._original_columns is not None

    @property
    def columns_in(self) -> pd.Index:
        if self._original_columns is None:
            raise RuntimeError("transformer not fitted")
        return self._original_columns

    @property
    @abstractmethod
    def columns_out(self) -> pd.Index:
        """
        :returns column labels for arrays returned by the fitted transformer
        """
        pass

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        :param deep If True, will return the parameters for this estimator and
        contained subobjects that are estimators

        :returns params Parameter names mapped to their values
        """
        # noinspection PyUnresolvedReferences
        return self.base_transformer.get_params(deep=deep)

    def set_params(self, **kwargs) -> "DataFrameTransformer":
        """
        Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        :returns self
        """
        # noinspection PyUnresolvedReferences
        self.base_transformer.set_params(**kwargs)
        return self

    # noinspection PyPep8Naming,PyUnusedLocal
    def _post_fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> None:
        self._original_columns = X.columns

    # noinspection PyPep8Naming
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> None:
        self._check_parameter_types(X, y)

        self._base_fit(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_parameter_types(X, None)

        transformed = self._base_transform(X)

        return pd.DataFrame(data=transformed, index=X.index, columns=self.columns_out)

    # noinspection PyPep8Naming
    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        self._check_parameter_types(X, y)

        transformed = self._base_fit_transform(X, y, **fit_params)

        self._post_fit(X, y, **fit_params)

        return pd.DataFrame(data=transformed, index=X.index, columns=self.columns_out)

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_parameter_types(X, None)

        transformed = self._base_inverse_transform(X)

        return pd.DataFrame(data=transformed, index=X.index, columns=self.columns_in)

    # noinspection PyPep8Naming
    def _base_fit(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params):
        # noinspection PyUnresolvedReferences
        self.base_transformer.fit(X, y, **fit_params)

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

    # noinspection PyPep8Naming
    @staticmethod
    def _check_parameter_types(X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("arg X must be a DataFrame")
        if y is not None and not isinstance(y, pd.Series):
            raise TypeError("arg y must be a Series")


class NumpyOnlyTransformer(
    DataFrameTransformer[_BaseTransformer], Generic[_BaseTransformer]
):
    """
    Special case of DataFrameTransformer where the base transformer does not accept
    data frames, but only numpy ndarrays
    """

    # noinspection PyPep8Naming
    def _base_fit(self, X: pd.DataFrame, y: Optional[pd.Series], **fit_params):
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


class ConstantColumnTransformer(
    DataFrameTransformer[_BaseTransformer], Generic[_BaseTransformer]
):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def columns_out(self) -> pd.Index:
        return self.columns_in


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

    @property
    def columns_out(self) -> pd.Index:
        column_transformer = self.base_transformer

        # construct the index from the columns in the fitted transformers
        return pd.Index(
            chain(
                *[
                    df_transformer.columns_out
                    for _, df_transformer, _ in column_transformer.transformers_
                    if df_transformer != "drop"
                ]
            )
        )


class SimpleImputerDF(DataFrameTransformer[SimpleImputer]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> SimpleImputer:
        return SimpleImputer(**kwargs)

    @property
    def columns_out(self) -> pd.Index:
        imputer = self.base_transformer

        return self.columns_in.delete(np.argwhere(np.isnan(imputer.statistics_)))


class OneHotEncoderDF(DataFrameTransformer[OneHotEncoder]):
    """
    A one-hot encoder that returns a DataFrame with correct row and column indices
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        encoder = self.base_transformer

        if encoder.sparse:
            raise ValueError(
                "sparse matrices not supported; set OneHotEncoder.sparse to False"
            )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> OneHotEncoder:
        return OneHotEncoder(**kwargs)

    @property
    def columns_out(self) -> pd.Index:
        encoder = self.base_transformer
        return pd.Index(encoder.get_feature_names(self.columns_in))


class OrdinalEncoderDF(ConstantColumnTransformer[OrdinalEncoder]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> OrdinalEncoder:
        return OrdinalEncoder(**kwargs)
