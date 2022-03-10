"""
Core implementation of :mod:`facet.validation`.
"""
import warnings
from abc import ABCMeta, abstractmethod
from typing import Generator, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from pytools.api import AllTracker

__all__ = [
    "BaseBootstrapCV",
    "BootstrapCV",
    "StratifiedBootstrapCV",
    "StationaryBootstrapCV",
]

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class BaseBootstrapCV(BaseCrossValidator, metaclass=ABCMeta):
    """
    Base class for bootstrap cross-validators.
    """

    def __init__(
        self,
        n_splits: int = 1000,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """
        :param n_splits: number of splits to generate (default: 1000)
        :param random_state: random state to initialise the random generator with
            (optional)
        """
        if n_splits < 1:
            raise ValueError(f"arg n_splits={n_splits} must be a positive integer")
        self.n_splits = n_splits
        self.random_state = random_state

    # noinspection PyPep8Naming
    def get_n_splits(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        groups: Sequence = None,
    ) -> int:
        """
        Return the number of splits generated by this cross-validator.

        :param X: for compatibility only, not used
        :param y: for compatibility only, not used
        :param groups: for compatibility only, not used
        :return: the number of splits
        """

        for arg_name, arg in ("X", X), ("y", y), ("groups", groups):
            if arg is not None:
                warnings.warn(
                    f"arg {arg_name} is not used but got {arg_name}={arg!r}",
                    stacklevel=2,
                )

        return self.n_splits

    # noinspection PyPep8Naming
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, pd.DataFrame, None] = None,
        groups: Union[np.ndarray, pd.Series, pd.DataFrame, None] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.

        :param X: features
        :param y: target: target variable for supervised learning problems,
            used as labels for stratification
        :param groups: ignored; exists for compatibility
        :return: a generator yielding `(train, test)` tuples where
            train and test are numpy arrays with train and test indices, respectively
        """

        n = len(X)

        if n < 2:
            raise ValueError("arg X must have at least 2 rows")

        if y is not None and n != len(y):
            raise ValueError("args X and y must have the same length")

        if groups is not None:
            warnings.warn(f"ignoring arg groups={groups!r}", stacklevel=2)

        rs = check_random_state(self.random_state)
        indices = np.arange(n)
        for i in range(self.n_splits):
            while True:
                train = self._select_train_indices(n_samples=n, random_state=rs, y=y)
                test_mask = np.ones(n, dtype=bool)
                test_mask[train] = False
                test = indices[test_mask]
                # make sure test is not empty, else sample another train set
                if len(test) > 0:
                    yield train, test
                    break

    @abstractmethod
    def _select_train_indices(
        self,
        n_samples: int,
        random_state: np.random.RandomState,
        y: Union[np.ndarray, pd.Series, pd.DataFrame, None],
    ) -> np.ndarray:
        """
        :param n_samples: number of indices to sample
        :param random_state: random state object to be used for random sampling
        :param y: labels for stratification
        :return: an array of integer indices with shape ``[n_samples]``
        """
        pass

    # noinspection PyPep8Naming
    def _iter_test_indices(self, X=None, y=None, groups=None) -> Iterator:
        # adding this stub just so all abstract methods are implemented
        pass


class BootstrapCV(BaseBootstrapCV):
    """
    Bootstrapping cross-validation.

    Generates CV splits by random sampling with replacement.
    The resulting training set is the same size as the total sample;
    the test set consists of all samples not included in the training set.

    Permissible as the ``cv`` argument of :class:`~sklearn.model_selection.GridSearchCV`
    object.
    """

    def _select_train_indices(
        self,
        n_samples: int,
        random_state: np.random.RandomState,
        y: Union[np.ndarray, pd.Series, pd.DataFrame, None],
    ) -> np.ndarray:
        return random_state.randint(n_samples, size=n_samples)


class StratifiedBootstrapCV(BaseBootstrapCV):
    """
    Stratified bootstrapping cross-validation.

    Generates CV splits by random sampling with replacement.
    The resulting training set is the same size as the total sample;
    the test set consists of all samples not included in the training set.

    Sampling is stratified based on a series or 1d array of group labels in the
    target vector.
    Bootstrapping is carried out separately for each group.
    """

    def _select_train_indices(
        self,
        n_samples: int,
        random_state: np.random.RandomState,
        y: Union[np.ndarray, pd.Series, pd.DataFrame, None],
    ) -> np.ndarray:
        if y is None:
            raise ValueError(
                "no target variable specified in arg y as labels for stratification"
            )
        if isinstance(y, pd.Series):
            y = y.values
        elif not (isinstance(y, np.ndarray) and y.ndim == 1):
            raise ValueError(
                "target labels must be provided as a Series or a 1d numpy array"
            )

        return (
            pd.Series(np.arange(len(y)))
            .groupby(by=y)
            .apply(
                lambda group: group.sample(
                    n=len(group), replace=True, random_state=random_state
                )
            )
            .values
        )


class StationaryBootstrapCV(BaseBootstrapCV):
    """
    Bootstrap for stationary time series, based on Politis and Romano (1994).

    This bootstrapping approach samples blocks with exponentially distributed sizes,
    instead of individual random observations as is the case with the regular bootstrap.

    Intended for use with time series that satisfy the stationarity requirement.

    """

    def __init__(
        self,
        n_splits: int = 1000,
        mean_block_size: Union[int, float] = 0.5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ) -> None:
        """
        :param n_splits: number of splits to generate (default: 1000)
        :param mean_block_size: mean size of coherent blocks to sample. If an ``int``,
            use this as the absolute number of blocks. If a ``float``, must be
            in the range (0.0, 1.0) and denotes a block size relative to the total
            number samples. (default: 0.5)
        :param random_state: random state to initialise the random generator with
            (optional)
        """
        super().__init__(n_splits=n_splits, random_state=random_state)
        if isinstance(mean_block_size, int):
            if mean_block_size < 2:
                raise ValueError(
                    f"arg mean_block_size={mean_block_size} must be at least 2"
                )
        elif isinstance(mean_block_size, float):
            if mean_block_size <= 0.0 or mean_block_size >= 1.0:
                raise ValueError(
                    f"arg mean_block_size={mean_block_size} must be > 0.0 and < 1.0"
                )
        else:
            raise TypeError(f"invalid type for arg mean_block_size={mean_block_size}")

        self.mean_block_size = mean_block_size

    def _select_train_indices(
        self,
        n_samples: int,
        random_state: np.random.RandomState,
        y: Union[np.ndarray, pd.Series, pd.DataFrame, None],
    ) -> np.ndarray:

        mean_block_size = self.mean_block_size
        if mean_block_size < 1:
            # if mean block size was set as a percentage, calculate the actual mean
            # block size
            mean_block_size = n_samples * mean_block_size

        p_new_block = 1.0 / mean_block_size

        train = np.empty(n_samples, dtype=np.int64)

        for i in range(n_samples):
            if i == 0 or random_state.uniform() <= p_new_block:
                idx = random_state.randint(n_samples)
            else:
                # noinspection PyUnboundLocalVariable
                idx += 1
                if idx >= n_samples:
                    idx = 0
            train[i] = idx

        return train


__tracker.validate()
