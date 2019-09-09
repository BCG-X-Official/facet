#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Cross-validation.

:class:`CircularCrossValidator` class performs cross-validation with a fixed
test_ratio with a fix size window shifting at a constant pace = 1/num_splits.
"""
from typing import *

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class CircularCrossValidator(BaseCrossValidator):
    """
    Rolling circular cross-validation.

    Class to generate various CV splits of train and test data sets using circular
    out-of-sample splits.

    Compatible with  :class:`sklearn.model_selection.GridSearchCV` object, if you set
    :code:`cv=circular_cross_validator`. See sklearn's `code
    <https://github.com/scikit-learn/scikit-learn/blob/7b136e9
    /sklearn/model_selection/_search.py#L961>`_.

    :param test_ratio:  Ratio determining the size of the test set (default=0.2).
    :param n_splits:   Number of splits to generate (default=50).
    """

    __slots__ = ["_test_ratio", "_n_splits", "_use_bootstrapping"]

    def __init__(self, test_ratio: float = 0.2, n_splits: int = 50) -> None:
        super().__init__()

        if not (0 < test_ratio < 1):
            raise ValueError(
                "Expected (0 < test_ratio < 1), but %d was given" % test_ratio
            )

        self._test_ratio = test_ratio
        self._n_splits = n_splits

    # noinspection PyPep8Naming
    def test_split_starts(self, X) -> Generator[int, None, None]:
        """
        Generate the start indices of the test splits.

        :param X: a feature matrix
        :return: generator of the first integer index of each test split
        """
        return (start for start, _ in self._test_split_bounds(self._n_samples(X)))

    def _test_split_bounds(
        self, n_samples: int
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Generate the start and end indices of the test splits.

        :param n_samples: number of samples
        :return: generator of the first and last integer index of each test split
        """
        step = n_samples / self._n_splits
        test_size = max(1.0, n_samples * self._test_ratio)
        for split in range(self._n_splits):
            split_start = split * step
            yield (int(split_start), int(split_start + test_size))

    # noinspection PyPep8Naming
    @staticmethod
    def _n_samples(X=None, y=None) -> int:
        """
        Return the number of samples.

        :return: the number of samples in X and y
        """
        if X is not None:
            if y is not None and len(X) != len(y):
                raise ValueError("X and y must be the same length")
            return len(X)
        elif y is not None:
            return len(y)
        else:
            raise ValueError("Need to specify at least one of X or y")

    # noinspection PyPep8Naming
    def _iter_test_indices(
        self, X=None, y=None, groups=None
    ) -> Generator[np.array, None, None]:
        """
        Generate the indices of the test splits.

        Generator which yields the numpy arrays of the test_split indices.

        :param X: features (need to speficy if y is None)
        :param y: targets (need to specify if X is None)
        :param groups: not used in this implementation, which is solely based on
          num_samples, num_splits, test_ratio
        :return: Iterable (Generator of numpy arrays) of all test-sets
        """
        n_samples = self._n_samples(X, y)

        data_indices = np.arange(n_samples)

        for test_start, test_end in self._test_split_bounds(n_samples):
            data_indices_rolled = np.roll(data_indices, -test_start)
            test_indices = data_indices_rolled[: test_end - test_start]
            yield test_indices

    # noinspection PyPep8Naming
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Return the number of splits.

        Implementation of method in BaseCrossValidator: returns the number of splits

        :param X: not used in this implementation, which is solely based on
          num_samples, num_splits, test_ratio
        :param y: not used in this implementation, which is solely based on
          num_samples, num_splits, test_ratio
        :param groups: not used in this implementation, which is solely based on
          num_samples, num_splits, test_ratio
        :return: Returns the number of splits as configured during the construction of
          the object
        """
        return self._n_splits
