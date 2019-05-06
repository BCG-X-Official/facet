from typing import *

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from yieldengine import deprecated


class CircularCrossValidator(BaseCrossValidator):
    """
    Class to generate various CV folds of train and test datasets using circular out-of-sample splits.

    Compatible with scikit-learn's GridSearchCV object, if you set :code:`cv=circular_cross_validator`

    See scikit-learn's `code <https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_search.py#L961>`_
    and `reference <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

    Unless you call :code:`resample()`, **this class will keep drawn folds stable, even if use_bootstrapping = True**

    :param num_samples: Number of samples of the input dataset - needed to give deterministic folds on multiple\
     call of get_train_test_splits_as_indices
    :param test_ratio:  Ratio determining the size of the test set (default=0.2).
    :param num_folds:   Number of folds to generate (default=50).
    :param use_bootstrapping: Whether to bootstrap samples (default=False)

    """

    def __init__(self, test_ratio: float = 0.2, num_folds: int = 50,
                 use_bootstrapping: bool = False) -> None:
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
                int(fold * step)
                for fold in range(self.__num_folds)
            ]

        self.__splits_defined = True

    def _generate_train_test_splits_as_indices(
            self, n_samples
    ) -> Generator[Tuple[np.array, np.array], None, None]:
        """
        Retrieves all generated folds of (train, test) pairs as tuples of arrays with the indices

        Meant to be used as "cv" parameter i.e. for scikit-learn's GridSearchCV

        :return: A generator of tuples of kind (ndarray, ndarray). If you need a list, simply \
        call :code:`list(circular_cross_validator.get_train_test_splits_as_indices(...))`
        """

        # ensure splits have been defined:
        if not self.__splits_defined:
            self.__define_splits(n_samples)

        data_indices = np.arange(n_samples)

        n_test_samples = max(1, int(n_samples * self._test_ratio))

        for fold_test_start_sample in self.__test_splits_start_samples:
            data_indices_rolled = np.roll(data_indices, fold_test_start_sample)
            test_indices = data_indices_rolled[0: n_test_samples]
            train_indices = data_indices_rolled[n_test_samples:]
            # conform to scikit-learn, expecting " - An iterable yielding (train, test) splits as arrays of indices."
            # see: https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_search.py#L961
            yield (train_indices, test_indices)

    @deprecated("to be moved to separate class")
    def get_train_test_splits_as_dataframes(
            self, input_dataset: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Retrieves all generated folds of (train, test) pairs as tuples of dataframes

        :param input_dataset: A pd.DataFrame object containing all data to split.
        :return: A generator of tuples of kind (pd.DataFrame, pd.DataFrame). If you need a list, simply \
        call :code:`list(circular_cross_validator.get_train_test_splits_as_dataframes(...))`
        """
        if input_dataset is None or not type(input_dataset) == pd.DataFrame:
            raise ValueError("Expected a pandas.DataFrame as input_dataset")

        for (
                train_indices,
                test_indices,
        ) in self._generate_train_test_splits_as_indices(len(input_dataset)):
            yield (
                (input_dataset.iloc[train_indices], input_dataset.iloc[test_indices])
            )

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

        for (
                train_indices,
                test_indices,
        ) in self._generate_train_test_splits_as_indices(len(X)):
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
