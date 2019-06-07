import copy
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from yieldengine import Sample
from yieldengine.model import Model, PipelineDF


class PredictorCV:
    __slots__ = [
        "_model",
        "_cv",
        "_sample",
        "_predictions_for_all_samples",
        "_model_by_fold",
    ]

    F_FOLD_ID = "fold_id"
    F_PREDICTION = "prediction"
    F_TARGET = "target"

    def __init__(self, model: Model, cv: BaseCrossValidator, sample: Sample) -> None:
        self._model = model
        self._cv = cv
        self._sample = sample
        self._model_by_fold: Optional[Dict[int, Model]] = None
        self._predictions_for_all_samples: Optional[pd.DataFrame] = None

    @property
    def cv(self) -> BaseCrossValidator:
        return self._cv

    @property
    def sample(self) -> Sample:
        return self._sample

    @property
    def model_by_fold(self) -> Optional[Dict[int, Model]]:
        return self._model_by_fold

    def pipeline(self, fold: int) -> PipelineDF:
        """
        :param fold: start index of test fold
        :return: the pipeline that was used to predict the dependent variable of
        the test fold
        """
        if self._model_by_fold is None:
            self.predictions_for_all_samples()
        return self._model_by_fold[fold].pipeline()

    def predictions_for_fold(self, fold_id: int) -> pd.Series:
        all_predictions = self.predictions_for_all_samples()
        return all_predictions.loc[
            all_predictions[PredictorCV.F_FOLD_ID] == fold_id, PredictorCV.F_PREDICTION
        ]

    def estimator(self, fold: int) -> BaseEstimator:
        """
        :param fold: start index of test fold
        :return: the estimator that was used to predict the dependent variable of
        the test fold
        """
        if self._model_by_fold is None:
            self.predictions_for_all_samples()
        return self._model_by_fold[fold].estimator

    def _is_fitted(self) -> bool:
        return self._model_by_fold is not None

    def _fit(self) -> None:
        self._model_by_fold: Dict[int, Pipeline] = {}

        sample = self.sample

        for fold_id, (train_indices, _) in enumerate(
            self.cv.split(sample.features, sample.target)
        ):
            train_sample = sample.select_observations(numbers=train_indices)

            self._model_by_fold[fold_id] = model = self._model.clone()

            pipeline = model.pipeline()

            pipeline.fit(X=train_sample.features, y=train_sample.target)

    def predictions_for_all_samples(self) -> pd.DataFrame:
        """
        For each fold of this Predictor's CV, predict all
        values in the test set. The result is a data frame with one row per
        prediction, indexed by the observations in the sample, and with columns
        F_FOLD_ID (the numerical index of the start of the test set in the current
        fold), F_PREDICTION (the predicted value for the given observation and fold),
        and F_TARGET (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        folds overlap.

        :return: the data frame with the predictions per observation and test fold
        """

        if self._predictions_for_all_samples is not None:
            return self._predictions_for_all_samples

        if not self._is_fitted():
            self._fit()

        sample = self.sample

        def predict(fold_id: int, test_indices: np.ndarray) -> pd.DataFrame:
            test_sample = sample.select_observations(numbers=test_indices)

            pipeline = self.pipeline(fold=fold_id)

            return pd.DataFrame(
                data={
                    PredictorCV.F_FOLD_ID: fold_id,
                    PredictorCV.F_PREDICTION: pipeline.predict(X=test_sample.features),
                },
                index=test_sample.index,
            )

        self._predictions_for_all_samples = pd.concat(
            [
                predict(fold_id, test_indices=test_indices)
                for fold_id, (_, test_indices) in enumerate(
                    self.cv.split(sample.features, sample.target)
                )
            ]
        ).join(sample.target.rename(PredictorCV.F_TARGET))

        return self._predictions_for_all_samples

    def copy_with_sample(self, sample: Sample):
        copied_predictor = copy.copy(self)
        copied_predictor._sample = sample
        return copied_predictor
