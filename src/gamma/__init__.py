#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, even not in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Global definitions of basic types and functions for use across the yield engine library
"""

import logging
from copy import copy
from functools import wraps
from typing import *

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# noinspection PyShadowingBuiltins
_T = TypeVar("_T")
ListLike = Union[np.ndarray, pd.Series, Sequence[_T]]
MatrixLike = Union[
    np.ndarray, pd.Series, pd.DataFrame, Sequence[Union[_T, "MatrixLike[_T]"]]
]


def deprecated(message: str):
    """
    Decorator to mark functions as deprecated.

    It will result in a warning being logged when the function is used.

    :return: decorator; the decorated functions logs a warning message saying it is
      deprecated
    """

    def _deprecated_inner(func: callable) -> callable:
        @wraps(func)
        def new_func(*args, **kwargs) -> Any:
            """
            Function wrapper
            """
            message_header = f"Call to deprecated function {func.__name__}"
            if message is None:
                log.warning(message_header)
            else:
                log.warning(f"{message_header}: {message}")
            return func(*args, **kwargs)

        return new_func

    return _deprecated_inner


class Sample:
    """
    Utility class to wrap a Pandas DataFrame in order to easily access its

        - features as a data frame
        - target as a series
        - feature columns by type, e.g., numbers or objects

    via object properties.

    An added benefit is through several checks:

        - features & target columns need to be defined explicitly
        - target column is not allowed as part of the features

    """

    DTYPE_NUMERICAL = pd.np.number
    DTYPE_OBJECT = object
    DTYPE_DATETIME = pd.np.datetime64
    DTYPE_TIMEDELTA = pd.np.timedelta64
    DTYPE_CATEGORICAL = "category"
    DTYPE_DATETIME_TZ = "datetimetz"

    __slots__ = ["_observations", "_target_name", "_feature_names"]

    def __init__(
        self,
        observations: pd.DataFrame,
        target_name: str,
        feature_names: ListLike[str] = None,
    ) -> None:
        """
        Construct a Sample object.

        :param observations: a Pandas DataFrame
        :param target_name: string of column name that constitutes as the target
        variable
        :param feature_names: iterable of column names that constitute as feature
        variables or \
        ``None``, in which case all non-target columns are features
        """
        if observations is None or not isinstance(observations, pd.DataFrame):
            raise ValueError("sample is not a DataFrame")

        self._observations = observations

        if target_name is None or not isinstance(target_name, str):
            raise KeyError("target is not a string")

        if target_name not in self._observations.columns:
            raise KeyError(
                f"target '{target_name}' is not a column in the observations table"
            )

        self._target_name = target_name

        if feature_names is None:
            feature_names = observations.columns.drop(labels=self._target_name)
        else:
            # check if all provided feature names actually exist in the observations df
            missing_columns = [
                name
                for name in feature_names
                if not observations.columns.contains(key=name)
            ]
            if len(missing_columns) > 0:
                missing_columns_list = '", "'.join(missing_columns)
                raise KeyError(
                    "observations table is missing columns for features "
                    f'"{missing_columns_list}"'
                )

            # ensure target column is not part of features:
            if self._target_name in feature_names:
                raise KeyError(
                    f"features include the target column {self._target_name}"
                )

        self._feature_names = feature_names

    @property
    def target_name(self) -> str:
        """Name of the target column."""
        return self._target_name

    @property
    def feature_names(self) -> ListLike[str]:
        """List of feature column names."""
        return self._feature_names

    @property
    def index(self) -> pd.Index:
        """Index of all observations in this sample."""
        return self.target.index

    @property
    def target(self) -> pd.Series:
        """
        :return: the target column as a series
        """
        return self._observations.loc[:, self._target_name]

    @property
    def features(self) -> pd.DataFrame:
        """
        :return: all feature columns as a data frame
        """
        return self._observations.loc[:, self._feature_names]

    def features_by_type(
        self, dtype: Union[type, str, Sequence[Union[type, str]]]
    ) -> pd.DataFrame:
        """
        Return a data frame with columns for all features matching the given type

        :param dtype: dtype, or sequence of dtypes, for filtering features.
          See DTYPE_*constants for common type selectors
        :return: data frame of the selected features
        """
        return self.features.select_dtypes(dtype)

    def select_observations_by_position(
        self, positions: Optional[Iterable[int]] = None
    ) -> "Sample":
        """
        Select observations by positional indices (`iloc`)
        :param positions: positional indices of observations to select
        :return: copy of this sample, containing only the observations at the given
        indices
        """
        subsample = copy(self)
        subsample._observations = self._observations.iloc[positions, :]
        return subsample

    def select_observations_by_index(self, ids: Iterable[Any] = None) -> "Sample":
        """
        Select observations index items (`loc`)

        :param ids: indices of observations to select
        :return: copy of this sample, containing only the observations at the given
          indices
        """
        subsample = copy(self)
        subsample._observations = self._observations.loc[ids, :]
        return subsample

    def select_features(self, feature_names: ListLike[str]) -> "Sample":
        """
        Return a Sample object which only includes the given features

        :param feature_names: names of features to be selected
        :return: copy of this sample, containing only the features with the given names
        """
        subsample = copy(self)
        if not set(feature_names).issubset(self._feature_names):
            raise ValueError(
                "arg features is not a subset of the features in this sample"
            )
        subsample._feature_names = feature_names
        subsample._observations = self._observations.loc[
            :, [*feature_names, self._target_name]
        ]

        return subsample

    def __len__(self) -> int:
        return len(self._observations)
