"""
Core implementation of :mod:`gamma.ml`
"""

from copy import copy
from typing import *

import pandas as pd

from gamma.common import is_list_like


class Sample:
    """
    A set of observations comprising features, and one or more target variables.

    A `Sample` object is helpful to keep features and targets aligned and to keep
    ML code more readable. It provides basic methods for accessing features and targets,
    and for selecting subsets of features and observations.

    The underlying data structure is a pandas data frame.
    """

    __slots__ = ["_observations", "_target", "_features"]

    COL_OBSERVATION = "observation"
    COL_TARGET = "target"
    COL_FEATURE = "feature"

    def __init__(
        self,
        observations: pd.DataFrame,
        target: Union[str, Sequence[str]],
        features: Sequence[str] = None,
    ) -> None:
        """
        :param observations: the raw observed data as a pandas data frame
        :param target: one or more names of columns representing the target variable(s)
        :param features: optional sequence of strings naming the columns that \
            represent feature variables; if not stated then all non-target columns are
            considered features
        """

        def _ensure_columns_exist(column_type: str, columns: Iterable[str]):
            # check if all provided feature names actually exist in the observations df
            missing_columns = [
                name for name in columns if not observations.columns.contains(key=name)
            ]
            if len(missing_columns) > 0:
                missing_columns_list = '", "'.join(missing_columns)
                raise KeyError(
                    f"observations table is missing {column_type} columns "
                    f'{", ".join(missing_columns_list)}'
                )

        if observations is None or not isinstance(observations, pd.DataFrame):
            raise ValueError("arg observations is not a DataFrame")

        index = observations.index

        if index.nlevels != 1:
            raise ValueError(
                f"arg observations has an index with {index.nlevels} levels, but "
            )

        # make sure the index has a name (but don't overwrite the original observations
        # data frame
        if index.name is None:
            observations = observations.copy(deep=False)
            observations.index = index.rename(Sample.COL_OBSERVATION)

        self._observations = observations

        multi_target = is_list_like(target)

        if multi_target:
            _ensure_columns_exist(column_type="target", columns=target)
        else:
            if target not in self._observations.columns:
                raise KeyError(
                    f'target "{target}" is not a column in the observations table'
                )

        self._target = target

        if features is None:
            features = observations.columns.drop(labels=target)
        else:
            _ensure_columns_exist(column_type="feature", columns=features)

            # ensure features and target(s) do not overlap

            if multi_target:
                shared = set(target).intersection(features)
                if len(shared) > 0:
                    raise KeyError(
                        f'targets {", ".join(shared)} are also included in the features'
                    )
            else:
                if target in features:
                    raise KeyError(f"target {target} is also included in the features")

        self._features = features

    @property
    def index(self) -> pd.Index:
        """Row index of all observations in this sample."""
        return self.target.index

    @property
    def target(self) -> Union[pd.Series, pd.DataFrame]:
        """
        The target variables for all observations.

         Returned as a series if there is only a single target, or as a data frame if
         there are multiple targets
        """
        return self._observations.loc[:, self._target]

    @property
    def features(self) -> pd.DataFrame:
        """
        The features for all observations.
        """
        return self._observations.loc[:, self._features]

    def subsample(
        self,
        *,
        loc: Optional[Union[slice, Sequence[Any]]] = None,
        iloc: Optional[Union[slice, Sequence[int]]] = None,
    ) -> "Sample":
        """
        Return a new sample with a subset of this sample's observations.

        Select observations either by indices (`loc` parameter), or integer indices
        (`iloc` parameter). Exactly one of both parameters must be provided when
        calling this method, not both or none.

        :param loc: indices of observations to select
        :param iloc: integer indices of observations to select
        :return: copy of this sample, comprising only the observations at the given \
            index locations
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

    def select_features(self, features: Sequence[str]) -> "Sample":
        """
        Return a new sample which only includes the features with the given names.

        :param features: names of the features to be selected
        :return: copy of this sample, containing only the features with the given names
        """
        if not set(features).issubset(self._features):
            raise ValueError(
                "arg features is not a subset of the features in this sample"
            )

        subsample = copy(self)
        subsample._features = features

        target = self._target
        if not is_list_like(target):
            target = [target]

        subsample._observations = self._observations.loc[:, [*features, *target]]

        return subsample

    def replace_features(self, features: pd.DataFrame) -> "Sample":
        """
        Return a new sample using the given features, and the target variable(s) of \
        this sample.

        The index of the `features` argument must be a subset of, or equal to, the row \
        index of this sample.
        :param features: the features to use for the new sample
        :return: the resulting, new sample object
        """
        target = self.target

        if not features.index.isin(self.index).all():
            raise ValueError(
                "index of arg features contains items that do not exist in this sample"
            )

        return Sample(
            observations=features.join(target),
            target=target.name if isinstance(target, pd.Series) else target.columns,
        )

    def __len__(self) -> int:
        """
        :return: the number of observations in this sample
        """
        return len(self._observations)
