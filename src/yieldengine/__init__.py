import logging
from copy import copy
from typing import *

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def deprecated(message: str):
    """
    Decorator to mark functions as deprecated.
    It will result in a warning being logged when the function is used.
    """

    def _deprecated_inner(func: callable) -> callable:
        def new_func(*args, **kwargs) -> Any:
            """
            Function wrapper
            """
            message_header = "Call to deprecated function {}".format(func.__name__)
            if message is None:
                log.warning(message_header)
            else:
                log.warning("{}: {}".format(message_header, message))
            return func(*args, **kwargs)

        return new_func

    return _deprecated_inner


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

    __slots__ = ["_observations", "_target_name", "_feature_names"]

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

        self._feature_names = feature_names_set

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
        return self._feature_names

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
        return self._observations.loc[:, self._feature_names]

    def features_by_type(
        self, dtype: Union[type, str, Sequence[Union[type, str]]]
    ) -> pd.DataFrame:
        """
        :param dtype: dtype, or sequence of dtypes, for filtering features. See DTYPE_*
        constants for common type selectors
        :return: list of columns for filtered features
        """
        return self.features.select_dtypes(dtype)

    def select_observations(
        self,
        numbers: Optional[Iterable[int]] = None,
        ids: Optional[Iterable[Any]] = None,
    ) -> "Sample":
        """
        Select observations either by numerical indices (`iloc`) or index items (`loc`)
        :param numbers: numerical indices of observations to select (optional)
        :param ids: indices of observations to select (optional)
        :return: copy of this sample, containing only the observations at the given
        indices
        """
        if numbers is None:
            if ids is None:
                raise ValueError(
                    "need to specify either numbers or ids to select observations by"
                )
        elif ids is not None:
            raise ValueError(
                "need to specify only one of either numbers or ids to select observations by"
            )

        subsample = copy(self)

        if numbers is not None:
            subsample._observations = self._observations.iloc[numbers, :]
        else:
            subsample._observations = self._observations.loc[ids, :]

        return subsample

    def select_features(self, feature_names: Iterable[str]) -> "Sample":
        subsample = copy(self)
        feature_names = list(feature_names)
        feature_set = set(feature_names)
        if not feature_set.issubset(self._feature_names):
            raise ValueError(
                "arg features is not a subset of the features in this sample"
            )
        subsample._observations = self._observations.loc[
            :, [*feature_names, subsample.target_name]
        ]
        subsample._feature_names = feature_set

        return subsample

    def observed_feature_values(
        self,
        feature_name: str,
        min_relative_frequency: float = 0.05,
        limit_observations: int = 20,
    ) -> np.ndarray:
        """
        Get an array of observed values for a particular feature

        :param feature_name: name of the feature
        :param min_relative_frequency: the relative frequency with which a particular
        feature value has to occur within the sample, for it to be selected. Not used
        for non-discrete features or features with high variability (when no single
        feature value occurs more than "min_relative_frequency" times)
        :param limit_observations: how many observation-values to return at max.
        :return: a 1D numpy array with the selected feature values
        """

        # get the series of the feature and drop NAs
        feature_series = self._observations.loc[:, feature_name].dropna()

        # get value counts
        times_observed = feature_series.value_counts()

        # get relative frequency for each feature value and filter using
        # min_relative_frequency, then limit using limit_observations
        observed_filtered = (
            times_observed[
                times_observed / sum(times_observed) >= min_relative_frequency
            ]
            .index[:limit_observations]
            .to_numpy()
        )

        # feature is categorical or object? -> always only use frequency based approach
        if (
            feature_name
            in self._observations.select_dtypes(
                [Sample.DTYPE_OBJECT, Sample.DTYPE_CATEGORICAL]
            ).columns
        ):
            return observed_filtered

        # feature is numeric and either
        #  a) feature is non-discrete/non-int datatype
        #  b) above approach did not return any feature values (i.e. because of too
        #  much variation even in an all integer feature)
        # --> go with approach below
        if len(observed_filtered) == 0 or (
            # not all values of the series convert well to int datatype without changes
            not np.all(feature_series == feature_series.astype(int))
        ):
            # get a sorted array of all unique values for the feature
            unique_values = np.asarray(sorted(feature_series.unique()))
            # are there more unique-values than allowed by the passed limit?
            if len(unique_values) > limit_observations:
                # use np.linspace to spread out array indices evenly within bounds
                value_samples = np.linspace(
                    0, len(unique_values) - 1, limit_observations
                ).astype(int)
                # return "sampled" feature values out of all unique feature values
                return unique_values[value_samples]
            else:
                # return all unique values, since they are within limit bound
                return unique_values
        else:
            return observed_filtered

    # Note: importing "DataFrameTransformer" in this module creates a
    # cycle...hence, use a quoted type-hint and ignore the inspection:
    # noinspection PyUnresolvedReferences
    def transformed(self, transformer: "DataFrameTransformer") -> "Sample":
        return transformer.fit_transform_sample(sample=self)

    def __len__(self) -> int:
        return len(self._observations)
