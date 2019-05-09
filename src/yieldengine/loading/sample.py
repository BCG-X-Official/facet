from copy import copy
from typing import *

import numpy as np
import pandas as pd


class Sample:
    """
    Utility class to wrap a Pandas DataFrame in order to easily access its

        - features as a dataframe
        - target as a series
        - feature columns by type, e.g., numbers or objects

    via object properties.

    An added benefit is through several checks:

        - features & target columns need to be defined explicitly
        - target column is not allowed as part of the features

    """

    DTYPE_NUMERICAL = np.number
    DTYPE_OBJECT = object
    DTYPE_DATETIME = np.datetime64
    DTYPE_TIMEDELTA = np.timedelta64
    DTYPE_CATEGORICAL = "category"
    DTYPE_DATETIME_TZ = "datetimetz"

    __slots__ = ["_observations", "_target_name", "_features_names"]

    def __init__(
        self,
        observations: pd.DataFrame,
        target_name: str,
        feature_names: Iterable[str] = None,
    ) -> None:
        """
        Construct a Sample object.

        :param observations: a Pandas DataFrame
        :param target_name: string of column name that constitutes as the target
        variable
        :param feature_names: iterable of column names that constitute as feature
        variables or \
        None, in which case all non-target columns are features
        """
        if observations is None or not isinstance(observations, pd.DataFrame):
            raise ValueError("sample is not a DataFrame")

        self._observations = observations

        if target_name is None or not isinstance(target_name, str):
            raise ValueError("target is not a string")

        if target_name not in self._observations.columns:
            raise ValueError(
                f"target '{target_name}' is not a column in the observations table"
            )

        self._target_name = target_name

        if feature_names is None:
            feature_names_set = {
                c for c in observations.columns if c != self._target_name
            }
        else:
            feature_names_set: Set[str] = set(feature_names)

        if not feature_names_set.issubset(observations.columns):
            missing_columns = feature_names_set.difference(observations.columns)
            raise ValueError(
                "observations table is missing columns for some features: "
                f"{missing_columns}"
            )
        # ensure target column is not part of features:
        if self._target_name in feature_names_set:
            raise ValueError(f"features includes the target column {self._target_name}")

        self._features_names = feature_names_set

    @property
    def target_name(self) -> str:
        """
        :return: name of the target column
        """
        return self._target_name

    @property
    def feature_names(self) -> Collection[str]:
        """
        :return: list of feature column names
        """
        return self._features_names

    @property
    def index(self) -> pd.Index:
        """
        :return: index of all observations in this sample
        """
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
        return self._observations.loc[:, self._features_names]

    def features_by_type(self, dtype: Union[type, str]) -> Iterable[str]:
        """
        :param dtype: dtype for filtering features. See DTYPE_* constants for common
        type selectors
        :return: list of all numerical features
        """
        return self.features.select_dtypes(dtype).columns

    def select_observations(self, indices: Iterable[int]) -> "Sample":
        """
        :param indices: indices to select in the original sample
        :return: copy of this sample, containing only the observations at the given
        indices
        """
        subsample = copy(self)
        subsample._observations = self._observations.iloc[indices, :]
        return subsample

    def __len__(self) -> int:
        return len(self._observations)
