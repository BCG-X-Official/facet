"""
Core implementation of :mod:`facet`
"""

from copy import copy
from typing import Any, Collection, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

from pytools.api import AllTracker, is_list_like, to_list, to_set

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
    A collection of observations, comprising features as well as one or more target
    variables and optional sample weights.

    A :class:`.Sample` object serves to keep features, targets and weights aligned,
    thus keeping modeling code more readable and robust.
    It provides basic methods for accessing features, targets and weights, and
    for selecting subsets of features and observations.

    The underlying data structure is a pandas :class:`.DataFrame`.

    Supports :func:`.len`, returning the number of observations in this sample.
    """

    __slots__ = ["_observations", "_target", "_features", "_weight"]

    _observations: pd.DataFrame
    _weight: Optional[str]
    _features: List[str]
    _target: List[str]

    #: default name for the observations index (= row index)
    #: of the underlying data frame
    IDX_OBSERVATION = "observation"

    #: default name for the feature index (= column index)
    #: used when returning a features table
    IDX_FEATURE = "feature"

    #: default name for the target series or target index (= column index)
    #: used when returning the targets
    IDX_TARGET = "target"

    def __init__(
        self,
        observations: pd.DataFrame,
        *,
        target: Union[str, Sequence[str]],
        features: Optional[Sequence[str]] = None,
        weight: Optional[str] = None,
    ) -> None:
        """
        :param observations: a table of observational data; \
            each row represents one observation
        :param target: one or more names of columns representing the target variable(s)
        :param features: optional sequence of strings naming the columns that \
            represent features; if omitted, all non-target and non-weight columns are
            considered features
        :param weight: optional name of a column representing the weight of each \
            observation
        """

        def _ensure_columns_exist(column_type: str, columns: Iterable[str]):
            # check if all provided feature names actually exist in the observations df
            available_columns: pd.Index = observations.columns
            missing_columns = [
                name for name in columns if name not in available_columns
            ]
            if missing_columns:
                raise KeyError(
                    f"observations table is missing {column_type} columns "
                    f"{', '.join(missing_columns)}"
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

        target_list: List[str]

        multi_target = is_list_like(target)

        if multi_target:
            _ensure_columns_exist(column_type="target", columns=target)
            target_list = list(target)
        else:
            if target not in observations.columns:
                raise KeyError(
                    f'arg target="{target}" is not a column in the observations table'
                )
            target_list = [target]

        self._target = target_list

        # process the weight

        if weight is not None:
            if weight not in observations.columns:
                raise KeyError(
                    f'arg weight="{weight}" is not a column in the observations table'
                )

        self._weight = weight

        # process the features

        feature_list: List[str]

        if features is None:
            if weight is not None:
                _feature_index = observations.columns.drop(
                    labels=[*target_list, weight]
                )
            else:
                _feature_index = observations.columns.drop(labels=target_list)
            feature_list = _feature_index.to_list()
        else:
            _ensure_columns_exist(column_type="feature", columns=features)
            feature_list = list(features)

            # ensure features and target(s) do not overlap
            if multi_target:
                shared = set(target_list).intersection(feature_list)
                if len(shared) > 0:
                    raise KeyError(
                        f'targets {", ".join(shared)} are also included in the features'
                    )
            else:
                if target in features:
                    raise KeyError(f"target {target} is also included in the features")

        self._features = feature_list

        # make sure the index has a name

        if observations_index.name is None:
            observations = observations.rename_axis(index=Sample.IDX_OBSERVATION)

        # keep only the columns we need

        columns = [*feature_list, *target_list]
        if weight is not None and weight not in columns:
            columns.append(weight)

        self._observations = observations.loc[:, columns]

    @property
    def index(self) -> pd.Index:
        """
        Row index of all observations in this sample
        """
        return self._observations.index

    @property
    def feature_columns(self) -> List[str]:
        """
        The column names of all features in this sample
        """
        return self._features

    @property
    def target_columns(self) -> List[str]:
        """
        The column names of all targets in this sample
        """
        return self._target

    @property
    def weight_column(self) -> Optional[str]:
        """
        The column name of weights in this sample; ``None`` if no weights are defined.
        """
        return self._weight

    @property
    def features(self) -> pd.DataFrame:
        """
        The features for all observations
        """
        features: pd.DataFrame = self._observations.loc[:, self._features]

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
        target = self.target_columns

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
        if self._weight is not None:
            return self._observations.loc[:, self._weight]
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
        :return: copy of this sample, comprising only the observations in the given \
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

    def keep(self, features: Union[str, Collection[str]]) -> "Sample":
        """
        Return a new sample which only includes the features with the given names.

        :param features: name(s) of the features to be selected
        :return: copy of this sample, containing only the features with the given names
        """

        features: List[str] = to_list(features, element_type=str)

        if not set(features).issubset(self._features):
            raise ValueError(
                "arg features is not a subset of the features in this sample"
            )

        subsample = copy(self)
        subsample._features = features

        columns = [*features, *self._target]
        weight = self._weight
        if weight and weight not in columns:
            columns.append(weight)
        subsample._observations = self._observations.loc[:, columns]

        return subsample

    def drop(self, features: Union[str, Collection[str]]) -> "Sample":
        """
        Return a copy of this sample, dropping the features with the given names.

        :param features: name(s) of the features to be dropped
        :return: copy of this sample, excluding the features with the given names
        """
        features: Set[str] = to_set(features, element_type=str)

        unknown = features.difference(self._features)
        if unknown:
            raise ValueError(f"unknown features in arg features: {unknown}")

        return self.keep(
            features=[feature for feature in self._features if feature not in features]
        )

    def __len__(self) -> int:
        return len(self._observations)


__tracker.validate()
