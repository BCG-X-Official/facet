"""
Implementation of FACET's :class:`.Sample` class.
"""

import logging
from copy import copy
from typing import Any, Collection, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

from pytools.api import AllTracker, to_list, to_set

log = logging.getLogger(__name__)

__all__ = ["Sample"]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class Sample:
    """
    A collection of observations, comprising features, one or more target
    variables and optional sample weights.

    A :class:`.Sample` object serves to keep features, targets and weights aligned,
    ensuring a more readable and robust ML workflow.
    It provides basic methods for accessing features, targets and weights, and
    for selecting subsets of features and observations.

    The underlying data structure is a :class:`pandas.DataFrame`.

    Supports :func:`.len`, returning the number of observations in this sample.
    """

    __slots__ = ["_observations", "_target_names", "_feature_names", "_weight_name"]

    _observations: pd.DataFrame
    _weight_name: Optional[str]
    _feature_names: List[str]
    _target_names: List[str]

    #: Default name for the observations index (= row index)
    #: of the underlying data frame.
    IDX_OBSERVATION = "observation"

    #: Default name for the feature index (= column index)
    #: used when returning a features table.
    IDX_FEATURE = "feature"

    #: Default name for the target series or target index (= column index)
    #: used when returning the targets.
    IDX_TARGET = "target"

    def __init__(
        self,
        observations: pd.DataFrame,
        *,
        target_name: Union[str, Iterable[str]],
        feature_names: Optional[Iterable[str]] = None,
        weight_name: Optional[str] = None,
    ) -> None:
        """
        :param observations: a table of observational data;
            each row represents one observation
        :param target_name: the name of the column representing the target
            variable; or an iterable of names representing multiple targets
        :param feature_names: optional iterable of strings naming the columns that
            represent features; if omitted, all non-target and non-weight columns are
            considered features
        :param weight_name: optional name of a column representing the weight of each
            observation
        """

        def _ensure_columns_exist(column_type: str, columns: List[str]):
            # check if all provided feature names actually exist in the observations df
            available_columns: pd.Index = observations.columns
            missing_columns = {
                name for name in columns if name not in available_columns
            }
            if missing_columns:
                raise KeyError(
                    f"observations table is missing {column_type} columns "
                    f"{missing_columns}"
                )

        # check that the observations are valid

        if observations is None or not isinstance(observations, pd.DataFrame):
            raise ValueError("arg observations is not a DataFrame")

        observations_index = observations.index

        if observations_index.nlevels != 1:
            raise ValueError(
                f"index of arg observations has {observations_index.nlevels} levels, "
                "but is required to have 1 level"
            )

        # process the target(s)

        targets_list: List[str] = to_list(
            target_name, element_type=str, arg_name="target_name"
        )
        _ensure_columns_exist(column_type="target", columns=targets_list)

        self._target_names = targets_list

        # process the weight

        if weight_name is not None:
            if weight_name not in observations.columns:
                raise KeyError(
                    f'arg weight_name="{weight_name}" '
                    "is not a column in the observations table"
                )

        self._weight_name = weight_name

        # process the features

        features_list: List[str]

        if feature_names is None:
            if weight_name is not None:
                _feature_index = observations.columns.drop(
                    labels=[*targets_list, weight_name]
                )
            else:
                _feature_index = observations.columns.drop(labels=targets_list)
            features_list = _feature_index.to_list()
        else:
            features_list = to_list(
                feature_names, element_type=str, arg_name="feature_names"
            )
            _ensure_columns_exist(column_type="feature", columns=features_list)

            # ensure features and target(s) do not overlap
            shared = set(targets_list).intersection(features_list)
            if len(shared) > 0:
                raise KeyError(f"targets {shared} are also included in the features")

        self._feature_names = features_list

        # make sure the index has a name

        if observations_index.name is None:
            observations = observations.rename_axis(index=Sample.IDX_OBSERVATION)

        # keep only the columns we need

        observation_columns = [*features_list, *targets_list]
        if weight_name is not None and weight_name not in observation_columns:
            observation_columns.append(weight_name)

        self._observations = observations.loc[:, observation_columns]

    @property
    def index(self) -> pd.Index:
        """
        Row index of all observations in this sample.
        """
        return self._observations.index

    @property
    def feature_names(self) -> List[str]:
        """
        The column names of all features in this sample.
        """
        return self._feature_names

    @property
    def target_name(self) -> Union[str, List[str]]:
        """
        The column name of the target in this sample, or a list of column names
        if this sample has multiple targets.
        """
        if len(self._target_names) == 1:
            return self._target_names[0]
        else:
            return self._target_names

    @property
    def weight_name(self) -> Optional[str]:
        """
        The column name of weights in this sample; ``None`` if no weights are defined.
        """
        return self._weight_name

    @property
    def features(self) -> pd.DataFrame:
        """
        The features for all observations.
        """
        features: pd.DataFrame = self._observations.loc[:, self._feature_names]

        if features.columns.name is None:
            features = features.rename_axis(columns=Sample.IDX_FEATURE)

        return features

    @property
    def target(self) -> Union[pd.Series, pd.DataFrame]:
        """
        The target variable(s) for all observations.

        Represented as a series if there is only a single target, or as a data frame if
        there are multiple targets.
        """
        target = self._target_names

        if len(target) == 1:
            return self._observations.loc[:, target[0]]

        targets: pd.DataFrame = self._observations.loc[:, target]

        columns = targets.columns

        if columns.name is None:
            return targets.rename_axis(columns=Sample.IDX_TARGET)
        else:
            return targets

    @property
    def weight(self) -> Optional[pd.Series]:
        """
        A series indicating the weight for each observation; ``None`` if no weights
        are defined.
        """
        if self._weight_name is not None:
            return self._observations.loc[:, self._weight_name]
        else:
            return None

    def subsample(
        self,
        *,
        loc: Optional[Union[slice, Sequence[Any]]] = None,
        iloc: Optional[Union[slice, Sequence[int]]] = None,
    ) -> "Sample":
        """
        Return a new sample with a subset of this sample's observations.

        Select observations either by indices (``loc``), or integer indices
        (``iloc``). Exactly one of both arguments must be provided when
        calling this method, not both or none.

        :param loc: indices of observations to select
        :param iloc: integer indices of observations to select
        :return: copy of this sample, comprising only the observations in the given
            rows
        """
        subsample = copy(self)
        if iloc is None:
            if loc is None:
                ValueError("either arg loc or arg iloc must be specified")
            else:
                subsample._observations = self._observations.loc[loc, :]
        elif loc is None:
            subsample._observations = self._observations.iloc[iloc, :]
        else:
            raise ValueError(
                "arg loc and arg iloc must not both be specified at the same time"
            )
        return subsample

    def keep(self, *, feature_names: Union[str, Collection[str]]) -> "Sample":
        """
        Return a new sample which only includes the features with the given names.

        :param feature_names: name(s) of the features to be selected
        :return: copy of this sample, containing only the features with the given names
        """

        feature_names: List[str] = to_list(feature_names, element_type=str)

        if not set(feature_names).issubset(self._feature_names):
            raise ValueError(
                "arg feature_names is not a subset of the features in this sample"
            )

        subsample = copy(self)
        subsample._feature_names = feature_names

        columns = [*feature_names, *self._target_names]
        weight = self._weight_name
        if weight and weight not in columns:
            columns.append(weight)
        subsample._observations = self._observations.loc[:, columns]

        return subsample

    def drop(self, *, feature_names: Union[str, Collection[str]]) -> "Sample":
        """
        Return a copy of this sample, dropping the features with the given names.

        :param feature_names: name(s) of the features to be dropped
        :return: copy of this sample, excluding the features with the given names
        """
        feature_names: Set[str] = to_set(feature_names, element_type=str)

        unknown = feature_names.difference(self._feature_names)
        if unknown:
            raise ValueError(f"unknown features in arg feature_names: {unknown}")

        return self.keep(
            feature_names=[
                feature
                for feature in self._feature_names
                if feature not in feature_names
            ]
        )

    def __len__(self) -> int:
        return len(self._observations)


__tracker.validate()
