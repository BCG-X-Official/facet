from typing import *

import numpy as np
from sklearn.model_selection import BaseCrossValidator

class CircularCrossValidator(BaseCrossValidator):
    """
    Class to generate various CV folds of train and test datasets using circular out-of-sample splits.

    Compatible with scikit-learn's GridSearchCV object, if you set :code:`cv=circular_cross_validator`

    See scikit-learn's `code <https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_search.py#L961>`_
    and `reference <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

    :param test_ratio:  Ratio determining the size of the test set (default=0.2).
    :param num_folds:   Number of folds to generate (default=50).
    :param use_bootstrapping: Whether to bootstrap samples (default=False)

    """

    def __init__(
        self,
        test_ratio: float = 0.2,
        num_folds: int = 50,
        use_bootstrapping: bool = False,
    ) -> None:
        """
        :param test_ratio:  Ratio determining the size of the test set (default=0.2).
        :param num_folds:   Number of folds to generate (default=50).
        :param use_bootstrapping: Whether to bootstrap samples (default=False)
        :return: None
        """

        super().__init__()

        if not (0 < test_ratio < 1):
            raise ValueError(
                "Expected (0 < test_ratio < 1), but %d was given" % test_ratio
            )

        self._test_ratio = test_ratio
        self.__num_folds = num_folds
        self.__use_bootstrapping = use_bootstrapping
        self.__splits_defined = False

    def __define_splits(self, n_samples: int) -> None:
        """
        Function that defines splits, i.e. start-sample-index for each fold

        :return: None
        """

        if self.__use_bootstrapping:
            self.__test_splits_start_samples = np.random.randint(
                0, n_samples - 1, self.__num_folds
            )
        else:
            step = n_samples / self.__num_folds
            self.__test_splits_start_samples = [
                int(fold * step) for fold in range(self.__num_folds)
            ]

        self.__splits_defined = True

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

        if not self.__splits_defined:
            self.__define_splits(n_samples)

        data_indices = np.arange(n_samples)

        n_test_samples = max(1, int(n_samples * self._test_ratio))

        for fold_test_start_sample in self.__test_splits_start_samples:
            data_indices_rolled = np.roll(data_indices, fold_test_start_sample)
            test_indices = data_indices_rolled[0:n_test_samples]
            yield test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Implementation of method in BaseCrossValidator

        :param X: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :param y: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :param groups: not used in this implementation, which is solely based on num_samples, num_folds, test_ratio
        :return: Returns the number of folds as configured during the construction of the object
        """
        return self.__num_folds
