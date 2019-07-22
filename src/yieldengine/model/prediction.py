# coding=utf-8
"""
Fitted models with cross-validation.

:class:`PredictorFitCV` encapsulates a fully trained model.
It contains a :class:`.Model` (preprocessing + estimator), a dataset given by a
:class:`yieldengine.Sample` object and a
cross-validation method. The model is fitted accordingly.
"""
import copy
import logging
from abc import ABC
from enum import Enum
from typing import *

import pandas as pd
from joblib import delayed, Parallel
from sklearn.model_selection import BaseCrossValidator

from yieldengine import Sample
from yieldengine.model import Model
from yieldengine.prediction.classification import CalibratedClassifierCVDF

log = logging.getLogger(__name__)


class ProbabilityCalibrationMethod(Enum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class PredictorFitCV(ABC):
    """
    Full information about a model fitted with cross-validation.

    :param model: model to be fitted
    :param cv: the cross validator generating the train splits
    :param sample: the sample from which the training sets are drawn
    :param n_jobs: number of jobs to run in parallel. Default to ``None`` which is
      interpreted a 1.
    :param shared_memory: if ``True`` use threads in the parallel runs. If `False`
      use multiprocessing
    :param verbose: verbosity level used in the parallel computation
    """

    __slots__ = [
        "_model",
        "_cv",
        "_sample",
        "_n_jobs",
        "_shared_memory",
        "_verbose",
        "_model_by_split",
        "_predictions_for_all_samples",
    ]

    F_SPLIT_ID = "split_id"
    F_PREDICTION = "prediction"
    F_TARGET = "target"

    def __init__(
        self,
        model: Model,
        cv: BaseCrossValidator,
        sample: Sample,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        self._model = model
        self._cv = cv
        self._sample = sample
        self._n_jobs = n_jobs
        self._shared_memory = shared_memory
        self._verbose = verbose
        self._model_by_split: Optional[List[Model]] = None
        self._predictions_for_all_samples: Optional[pd.DataFrame] = None

    @property
    def model(self) -> Model:
        """The ingoing, usually unfitted model to be fitted to the training splits."""
        return self._model

    @property
    def cv(self) -> BaseCrossValidator:
        """The cross validator generating the train splits."""
        return self._cv

    @property
    def sample(self) -> Sample:
        """The sample from which the training sets are drawn."""
        return self._sample

    @property
    def n_splits(self) -> int:
        """Number of splits in this model fit."""
        return self.cv.get_n_splits(X=self.sample.features, y=self.sample.target)

    def fitted_models(self) -> Iterator[Model]:
        """Iterator of all models fitted for the train splits."""
        self._fit()
        return iter(self._model_by_split)

    def fitted_model(self, split_id: int) -> Model:
        """
        Return the fitted model for a given split.

        :param split_id: start index of test split
        :return: the model fitted for the train split at the given index
        """
        self._fit()
        return self._model_by_split[split_id]

    def _fit(self) -> None:

        if self._model_by_split is not None:
            return

        model = self.model
        sample = self.sample

        self._model_by_split: List[Model] = self._parrallel()(
            delayed(self._fit_model_for_split)(
                model.clone(),
                sample.select_observations_by_position(positions=train_indices),
            )
            for train_indices, _ in self.cv.split(sample.features, sample.target)
        )

    def _parrallel(self) -> Parallel:
        return Parallel(
            n_jobs=self._n_jobs,
            require="sharedmem" if self._shared_memory else None,
            verbose=self._verbose,
        )

    def _series_for_split(self, split_id: int, column: str) -> pd.Series:
        all_predictions: pd.DataFrame = self.predictions_for_all_splits()
        return all_predictions.xs(key=split_id, level=PredictorFitCV.F_SPLIT_ID).loc[
            :, column
        ]

    def predictions_for_split(self, split_id: int) -> pd.Series:
        """
        The predictions for a given split.

        :return: the series of predictions of the split
        """
        return self._series_for_split(
            split_id=split_id, column=PredictorFitCV.F_PREDICTION
        )

    def targets_for_split(self, split_id: int) -> pd.Series:
        """
        Return the target for this split.

        :return: the series of targets for this split"""
        return self._series_for_split(split_id=split_id, column=PredictorFitCV.F_TARGET)

    def predictions_for_all_splits(self) -> pd.DataFrame:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``F_SPLIT_ID``),
        and with columns ``F_PREDICTION` (the predicted value for the
        given observation and split), and ``F_TARGET`` (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the predictions per observation and test split
        """

        if self._predictions_for_all_samples is None:
            self._fit()

            sample = self.sample

            splitwise_predictions = []

            for split_id, (_, test_indices) in enumerate(
                self.cv.split(sample.features, sample.target)
            ):

                test_sample = sample.select_observations_by_position(
                    positions=test_indices
                )

                predictions = self.fitted_model(split_id=split_id).predict(
                    X=test_sample.features
                )

                predictions_df = pd.DataFrame(
                    data={
                        PredictorFitCV.F_SPLIT_ID: split_id,
                        PredictorFitCV.F_PREDICTION: predictions,
                    },
                    index=test_sample.index,
                )

                splitwise_predictions.append(predictions_df)

            self._predictions_for_all_samples = (
                pd.concat(splitwise_predictions)
                .join(sample.target.rename(PredictorFitCV.F_TARGET))
                .set_index(PredictorFitCV.F_SPLIT_ID, append=True)
            )

        return self._predictions_for_all_samples

    def copy_with_sample(self, sample: Sample):
        """
        Copy the predictor with some new :class:`yieldengine.Sample`.

        :param sample: the :class:`yieldengine.Sample` used for the copy
        :return: the copy of self
        """
        copied_predictor = copy.copy(self)
        copied_predictor._sample = sample
        copied_predictor._predictions_for_all_samples = None
        return copied_predictor

    @staticmethod
    def _fit_model_for_split(model: Model, train_sample: Sample) -> Model:
        """
        Fit a model using a sample.

        :param model:  the :class:`yieldengine.model.Model` to fit
        :param train_sample: data used to fit the model
        :return: fitted model for the split
        """
        model.fit(X=train_sample.features, y=train_sample.target)
        return model


class RegressorFitCV(PredictorFitCV):
    pass


class ClassifierFitCV(PredictorFitCV):
    __slots__ = [
        "_probabilities_for_all_samples",
        "_calibrated_model_by_split",
        "_calibration",
    ]

    F_PROBA = "proba_class_0"

    def __init__(
        self,
        model: Model,
        cv: BaseCrossValidator,
        sample: Sample,
        calibration: Optional[ProbabilityCalibrationMethod] = None,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            model=model,
            cv=cv,
            sample=sample,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

        self._calibration = calibration
        self._calibrated_model_by_split: Optional[List[Model]] = None
        self._probabilities_for_all_samples: Optional[pd.DataFrame] = None

    def probabilities_for_all_splits(self) -> pd.DataFrame:
        # todo: add support for multi-class classifiers
        # todo: add support for log probabilities
        if self._probabilities_for_all_samples is not None:
            return self._probabilities_for_all_samples

        self._fit()

        sample = self.sample

        splitwise_predictions = []

        for split_id, (_, test_indices) in enumerate(
            self.cv.split(sample.features, sample.target)
        ):
            test_sample = sample.select_observations_by_position(positions=test_indices)

            predictor = (
                self.fitted_model(split_id=split_id)
                if self._calibration is None
                else self.calibrated_model(split_id=split_id)
            )

            probabilities = predictor.predict_proba(X=test_sample.features)

            n_classes = probabilities.shape[1]

            # supporting only binary classification where n-classes == 2
            assert (
                n_classes == 2
            ), f"Got non-binary probabilities for {n_classes} classes"

            # just proceed with probabilities that it is class 0:
            probabilities = probabilities.loc[:, probabilities.columns[0]]

            predictions_df = pd.DataFrame(
                data={
                    PredictorFitCV.F_SPLIT_ID: split_id,
                    ClassifierFitCV.F_PROBA: probabilities,
                },
                index=test_sample.index,
            )

            splitwise_predictions.append(predictions_df)

        self._probabilities_for_all_samples = (
            pd.concat(splitwise_predictions)
            .join(sample.target.rename(PredictorFitCV.F_TARGET))
            .set_index(PredictorFitCV.F_SPLIT_ID, append=True)
        )

        return self._probabilities_for_all_samples

    def _fit(self) -> None:
        super()._fit()

        if self._calibration is not None:

            if self._calibrated_model_by_split is not None:
                return

            sample = self.sample
            log.info(
                f"Calibrating classifier probabilities"
                f" using method: {self._calibration.value}"
            )
            self._calibrated_model_by_split: List[Model] = self._parrallel()(
                delayed(self._calibrate_probabilities_for_split)(
                    # note: we specifically do not clone here, since
                    # CalibratedClassifierCV does expect a fitted classifier and does
                    # clone it itself - hence deepcopy so to be able to further
                    # differentiate between _model_by_split & _calibrated_model_by_split
                    self._model_by_split[idx],
                    sample.select_observations_by_position(positions=test_indices),
                    self._calibration,
                )
                for idx, (_, test_indices) in enumerate(
                    self.cv.split(sample.features, sample.target)
                )
            )

    @staticmethod
    def _calibrate_probabilities_for_split(
        model: Model, test_sample: Sample, calibration: ProbabilityCalibrationMethod
    ) -> Model:
        # todo: design more "elegant" approach to create new model w/o using deepcopy
        model = copy.deepcopy(model)

        cv = CalibratedClassifierCVDF(
            base_estimator=model.predictor, method=calibration.value, cv="prefit"
        )

        if model.preprocessing is not None:
            data_transformed = model.preprocessing.transform(test_sample.features)
        else:
            data_transformed = test_sample.features

        cv.fit(X=data_transformed, y=test_sample.target)

        model.predictor = cv.calibrated_classifiers_[0]

        return model

    def calibrated_model(self, split_id: int) -> Model:
        """
        :param split_id: start index of test split
        :return: the model fitted & calibrated for the train split at the given index
        """
        self._fit()
        return self._calibrated_model_by_split[split_id]

    def calibrated_models(self) -> Iterator[Model]:
        """
        :return: an iterator of all models fitted & calibrated over all train splits
        """
        if self._calibration is None:
            raise NotImplementedError("Calibration is 'None' for this ClassifierFitCV")

        self._fit()
        return iter(self._calibrated_model_by_split)
