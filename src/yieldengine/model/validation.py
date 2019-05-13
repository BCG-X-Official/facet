from typing import *

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class CircularCrossValidator(BaseCrossValidator):
    """
    Class to generate various CV folds of train and test datasets using circular
    out-of-sample splits.

    Compatible with scikit-learn's GridSearchCV object, if you set :code:`cv=circular_cross_validator`

    See scikit-learn's `code <https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_search.py#L961>`_
    and `reference <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

    :param test_ratio:  Ratio determining the size of the test set (default=0.2).
    :param num_folds:   Number of folds to generate (default=50).

    """

    __slots__ = ["_test_ratio", "_num_folds", "_use_bootstrapping"]

    def __init__(self, test_ratio: float = 0.2, num_folds: int = 50) -> None:
        """
        :param test_ratio:  Ratio determining the size of the test set (default=0.2).
        :param num_folds:   Number of folds to generate (default=50).
        :return: None
        """

        super().__init__()

        if not (0 < test_ratio < 1):
            raise ValueError(
                "Expected (0 < test_ratio < 1), but %d was given" % test_ratio
            )

        self._test_ratio = test_ratio
        self._num_folds = num_folds

    def _iter_test_indices(
        self, X=None, y=None, groups=None
    ) -> Generator[np.array, None, None]:
        """
        Implementation of method in BaseCrossValidator - yields iterable of indices of all test-sets

        :param X: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :param y: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :param groups: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :return: Iterable (Generator of np.arrays) of all test-sets
        """
        n_samples = len(X)

        def test_split_starts() -> Generator[int, None, None]:
            step = n_samples / self._num_folds
            for fold in range(self._num_folds):
                yield int(fold * step)

        data_indices = np.arange(n_samples)
        n_test_samples = max(1, int(n_samples * self._test_ratio))

        for fold_test_start_sample in test_split_starts():
            data_indices_rolled = np.roll(data_indices, fold_test_start_sample)
            test_indices = data_indices_rolled[0:n_test_samples]
            yield test_indices

    # noinspection PyPep8Naming
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Implementation of method in BaseCrossValidator

        :param X: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :param y: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :param groups: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :return: Returns the number of folds as configured during the construction of the object
        """
        return self._num_folds
