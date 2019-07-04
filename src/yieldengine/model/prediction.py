# coding=utf-8
"""Module with the `PredictorCV class which gather information from a model, its
cross validation and the sample used."""

import copy
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from yieldengine import Sample
from yieldengine.model import Model


class PredictorCV:
    """
    Class containing the information of a model that has been fitted with some \
    cross-validation.

    :param Model model: the underlying model
    :param BaseCrossValidator cv: underlying cross validator
    :param Sample sample: the `Sample` object used to fit the model.
    """
    __slots__ = [
        "_model",
        "_cv",
        "_sample",
        "_predictions_for_all_samples",
        "_model_by_split",
    ]

    F_SPLIT_ID = "split_id"
    F_PREDICTION = "prediction"
    F_TARGET = "target"

    def __init__(self, model: Model, cv: BaseCrossValidator, sample: Sample) -> None:
        self._model = model
        self._cv = cv
        self._sample = sample
        self._model_by_split: Optional[Dict[int, Model]] = None
        self._predictions_for_all_samples: Optional[pd.DataFrame] = None

    @property
    def cv(self) -> BaseCrossValidator:
        """The underlying `BaseCrossValidator`"""
        return self._cv

    @property
    def sample(self) -> Sample:
        """The underlying `Sample`"""
        return self._sample

    @property
    def model_by_split(self) -> Optional[Dict[int, Model]]:
        """
        When self is fitted, dictionary whit keys the split indexes (as int) and \
        values a clone of the fitted model for this split. `None` when not fitted.
        """
        return self._model_by_split

    @property
    def split_ids(self) -> Optional[Set[int]]:
        """The set of split ids when self is fitted, None otherwrise."""
        return set() if self.model_by_split is None else self.model_by_split.keys()

    def model(self, split_id: int) -> Model:
        """
        :param split_id: start index of test split
        :return: the model that was used to predict the dependent variable of \
        the test split
        """
        if self._model_by_split is None:
            self.predictions_for_all_samples()
        return self._model_by_split[split_id]

    def predictions_for_split(self, split_id: int) -> pd.Series:
        """Return the predictions for a given split.

        :return: Series of precictions for the split_id"""
        all_predictions = self.predictions_for_all_samples()
        return all_predictions.loc[
            all_predictions[PredictorCV.F_SPLIT_ID] == split_id,
            PredictorCV.F_PREDICTION,
        ]

    def estimator(self, split: int) -> BaseEstimator:
        """
        :param split: start index of test split
        :return: the estimator that was used to predict the dependent variable of \
        the test split
        """
        if self._model_by_split is None:
            self.predictions_for_all_samples()
        return self._model_by_split[split].estimator

    def _is_fitted(self) -> bool:
        return self._model_by_split is not None

    def fit(self) -> None:
        """Fits the predictor."""
        if self._is_fitted():
            return

        self._model_by_split: Dict[int, Model] = {}

        sample = self.sample

        for split_id, (train_indices, _) in enumerate(
            self.cv.split(sample.features, sample.target)
        ):
            train_sample = sample.select_observations(numbers=train_indices)

            self._model_by_split[split_id] = model = self._model.clone()

            pipeline = model.pipeline()

            pipeline.fit(X=train_sample.features, y=train_sample.target)

    def predictions_for_all_samples(self) -> pd.DataFrame:
        """
        For each split of this Predictor's CV, predict all
        values in the test set. The result is a data frame with one row per
        prediction, indexed by the observations in the sample, and with columns
        F_SPLIT_ID (the numerical index of the start of the test set in the current
        split), F_PREDICTION (the predicted value for the given observation and split),
        and F_TARGET (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the predictions per observation and test split
        """

        if self._predictions_for_all_samples is not None:
            return self._predictions_for_all_samples

        if not self._is_fitted():
            self.fit()

        sample = self.sample

        def predict(split_id: int, test_indices: np.ndarray) -> pd.DataFrame:
            test_sample = sample.select_observations(numbers=test_indices)

            pipeline = self.model(split_id=split_id).pipeline()

            return pd.DataFrame(
                data={
                    PredictorCV.F_SPLIT_ID: split_id,
                    PredictorCV.F_PREDICTION: pipeline.predict(X=test_sample.features),
                },
                index=test_sample.index,
            )

        self._predictions_for_all_samples = pd.concat(
            [
                predict(split_id, test_indices=test_indices)
                for split_id, (_, test_indices) in enumerate(
                    self.cv.split(sample.features, sample.target)
                )
            ]
        ).join(sample.target.rename(PredictorCV.F_TARGET))

        return self._predictions_for_all_samples

    def copy_with_sample(self, sample: Sample):
        """
        Copies the predictor with some new `Sample`.

        :param sample: the `Sample` used for the copy
        :return: the copy of self
        """
        copied_predictor = copy.copy(self)
        copied_predictor._sample = sample
        copied_predictor._predictions_for_all_samples = None
        return copied_predictor
