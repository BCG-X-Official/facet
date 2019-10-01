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
Fitted predictions with cross-validation.

:class:`PredictionCV` encapsulates a fully trained pipeline.
It contains a :class:`.ModelPipelineDF` (preprocessing + estimator), a dataset given by a
:class:`yieldengine.Sample` object and a
cross-validation calibration. The pipeline is fitted accordingly.
"""
import copy
import logging
from abc import ABC
from typing import *

import pandas as pd
from joblib import delayed, Parallel
from sklearn.model_selection import BaseCrossValidator

from gamma.ml import Sample
from gamma.sklearndf import ClassifierDF, RegressorDF
from gamma.sklearndf.classification import CalibratedClassifierCVDF
from gamma.sklearndf.pipeline import (
    ClassifierPipelineDF,
    EstimatorPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)
from gamma.sklearndf.wrapper import ClassifierWrapperDF

log = logging.getLogger(__name__)

__all__ = ["PredictionCV", "RegressorPredictionCV", "ClassifierPredictionCV"]

_T_EstimatorPipelineDF = TypeVar("_T_EstimatorPipelineDF", bound=EstimatorPipelineDF)
_T_LearnerPipelineDF = TypeVar("_T_LearnerPipelineDF", bound=LearnerPipelineDF)
_T_ClassifierDF = TypeVar("_T_ClassifierDF", bound=ClassifierDF)
_T_RegressorDF = TypeVar("_T_RegressorDF", bound=RegressorDF)


class _BaseFitCV(ABC, Generic[_T_EstimatorPipelineDF]):
    """
    :class:~gamma.sklearn all splits of a given cross-validation
    strategy, based on a pipeline.

    :param pipeline: predictive pipeline to be fitted
    :param cv: the cross validator generating the train splits
    :param sample: the sample from which the training sets are drawn
    :param n_jobs: number of jobs to _rank_learners in parallel. Default to ``None`` which is
      interpreted a 1.
    :param shared_memory: if ``True`` use threads in the parallel runs. If `False`
      use multiprocessing
    :param verbose: verbosity level used in the parallel computation
    """

    __slots__ = [
        "_pipeline",
        "_cv",
        "_sample",
        "_n_jobs",
        "_shared_memory",
        "_verbose",
        "_model_by_split",
    ]

    def __init__(
        self,
        pipeline: _T_EstimatorPipelineDF,
        cv: BaseCrossValidator,
        sample: Sample,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        self._pipeline = pipeline
        self._cv = cv
        self._sample = sample
        self._n_jobs = n_jobs
        self._shared_memory = shared_memory
        self._verbose = verbose
        self._model_by_split: Optional[List[_T_EstimatorPipelineDF]] = None

    @property
    def pipeline(self) -> _T_EstimatorPipelineDF:
        """The ingoing, usually unfitted pipeline to be fitted to the training splits."""
        return self._pipeline

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
        """Number of splits in this pipeline fit."""
        return self.cv.get_n_splits(X=self.sample.features, y=self.sample.target)

    def __iter__(self) -> Iterator[_T_EstimatorPipelineDF]:
        """Iterator of all predictions fitted for the train splits."""
        self._fit()
        return iter(self._model_by_split)

    def __getitem__(self, split_id: int) -> _T_EstimatorPipelineDF:
        """
        Return the fitted pipeline for a given split.

        :param split_id: start index of test split
        :return: the pipeline fitted for the train split at the given index
        """
        self._fit()
        return self._model_by_split[split_id]

    def _fit(self) -> None:
        if self._model_by_split is not None:
            return

        pipeline = self.pipeline
        sample = self.sample

        self._model_by_split: List[_T_EstimatorPipelineDF] = self._parallel()(
            delayed(self._fit_model_for_split)(
                pipeline.clone(),
                sample.select_observations_by_position(positions=train_indices),
            )
            for train_indices, _ in self.cv.split(sample.features, sample.target)
        )

    def _parallel(self) -> Parallel:
        return Parallel(
            n_jobs=self._n_jobs,
            require="sharedmem" if self._shared_memory else None,
            verbose=self._verbose,
        )

    @staticmethod
    def _fit_model_for_split(
        pipeline: _T_EstimatorPipelineDF, train_sample: Sample
    ) -> _T_EstimatorPipelineDF:
        """
        Fit a pipeline using a sample.

        :param pipeline:  the :class:`gamma.ml.ModelPipelineDF` to fit
        :param train_sample: data used to fit the pipeline
        :return: fitted pipeline for the split
        """
        pipeline.fit(X=train_sample.features, y=train_sample.target)
        return pipeline


class PredictionCV(
    _BaseFitCV[_T_LearnerPipelineDF], Generic[_T_LearnerPipelineDF], ABC
):
    """
    Generate cross-validated predictions for each observation in a sample, based on
    multiple fits of a learner across a collection of cross-validation splits

    :param pipeline: predictive pipeline to be fitted
    :param cv: the cross validator generating the train splits
    :param sample: the sample from which the training sets are drawn
    :param n_jobs: number of jobs to _rank_learners in parallel. Default to ``None`` which is
      interpreted a 1.
    :param shared_memory: if ``True`` use threads in the parallel runs. If `False`
      use multiprocessing
    :param verbose: verbosity level used in the parallel computation
    """

    __slots__ = ["_predictions_for_all_samples"]

    COL_SPLIT_ID = "split_id"
    COL_PREDICTION = "prediction"
    COL_TARGET = "target"

    def __init__(
        self,
        pipeline: _T_LearnerPipelineDF,
        cv: BaseCrossValidator,
        sample: Sample,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            pipeline=pipeline,
            cv=cv,
            sample=sample,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

        self._predictions_for_all_samples: Optional[pd.DataFrame] = None

    def predictions_for_split(self, split_id: int) -> pd.Series:
        """
        The predictions for a given split.

        :return: the series of predictions of the split
        """
        return self._series_for_split(
            split_id=split_id, column=PredictionCV.COL_PREDICTION
        )

    def targets_for_split(self, split_id: int) -> pd.Series:
        """
        Return the target for this split.

        :return: the series of targets for this split"""
        return self._series_for_split(split_id=split_id, column=PredictionCV.COL_TARGET)

    def predictions_for_all_splits(self) -> pd.DataFrame:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``COL_SPLIT_ID``),
        and with columns ``COL_PREDICTION` (the predicted value for the
        given observation and split), and ``COL_TARGET`` (the actual target)

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

                predictions = self[split_id].predict(X=test_sample.features)

                predictions_df = pd.DataFrame(
                    data={
                        PredictionCV.COL_SPLIT_ID: split_id,
                        PredictionCV.COL_PREDICTION: predictions,
                    },
                    index=test_sample.index,
                )

                splitwise_predictions.append(predictions_df)

            self._predictions_for_all_samples = (
                pd.concat(splitwise_predictions)
                .join(sample.target.rename(PredictionCV.COL_TARGET))
                .set_index(PredictionCV.COL_SPLIT_ID, append=True)
            )

        return self._predictions_for_all_samples

    def copy_with_sample(self, sample: Sample):
        """
        Make a copy of this predictor using a new :class:`yieldengine.Sample`.

        :param sample: the :class:`yieldengine.Sample` used for the copy
        :return: the copy of self
        """
        copied_predictor = copy.copy(self)
        copied_predictor._sample = sample
        copied_predictor._predictions_for_all_samples = None
        return copied_predictor

    def _series_for_split(self, split_id: int, column: str) -> pd.Series:
        all_predictions: pd.DataFrame = self.predictions_for_all_splits()
        return all_predictions.xs(key=split_id, level=PredictionCV.COL_SPLIT_ID).loc[
            :, column
        ]


class RegressorPredictionCV(
    PredictionCV[RegressorPipelineDF[_T_RegressorDF]], Generic[_T_RegressorDF]
):
    pass


class ClassifierPredictionCV(
    PredictionCV[ClassifierPipelineDF[_T_ClassifierDF]], Generic[_T_ClassifierDF]
):
    __slots__ = ["_probabilities_for_all_samples", "_log_probabilities_for_all_samples"]

    COL_PROBA = "proba_class_0"

    CALIBRATION_SIGMOID = "sigmoid"
    CALIBRATION_ISOTONIC = "isotonic"

    def __init__(
        self,
        pipeline: ClassifierPipelineDF[_T_ClassifierDF],
        cv: BaseCrossValidator,
        sample: Sample,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            pipeline=pipeline,
            cv=cv,
            sample=sample,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

        self._probabilities_for_all_samples: Optional[pd.DataFrame] = None
        self._log_probabilities_for_all_samples: Optional[pd.DataFrame] = None

    def probabilities_for_all_splits(self) -> pd.DataFrame:
        if self._probabilities_for_all_samples is None:
            self._probabilities_for_all_samples = self._probabilities_for_all_splits(
                log_proba=False
            )

        return self._probabilities_for_all_samples

    def log_probabilities_for_all_splits(self) -> pd.DataFrame:
        if self._log_probabilities_for_all_samples is None:
            self._log_probabilities_for_all_samples = self._probabilities_for_all_splits(
                log_proba=True
            )

        return self._log_probabilities_for_all_samples

    def _pipeline_for_split(self, split_id: int) -> ClassifierPipelineDF:
        return self[split_id]

    def _probabilities_for_all_splits(self, log_proba: bool) -> pd.DataFrame:
        self._fit()

        sample = self.sample

        predictions_per_split = []

        for split_id, (_, test_indices) in enumerate(
            self.cv.split(sample.features, sample.target)
        ):
            test_sample = sample.select_observations_by_position(positions=test_indices)

            pipeline = self._pipeline_for_split(split_id)

            if log_proba:
                probabilities = pipeline.predict_log_proba(X=test_sample.features)
            else:
                probabilities = pipeline.predict_proba(X=test_sample.features)

            predictions_df = probabilities.join(
                pd.Series(
                    data=split_id,
                    index=probabilities.index,
                    name=PredictionCV.COL_SPLIT_ID,
                )
            )

            predictions_per_split.append(predictions_df)

        return (
            pd.concat(predictions_per_split)
            .join(sample.target.rename(PredictionCV.COL_TARGET))
            .set_index(PredictionCV.COL_SPLIT_ID, append=True)
        )


class CalibratedClassifierPredictionCV(
    ClassifierPredictionCV[_T_ClassifierDF], Generic[_T_ClassifierDF]
):
    __slots__ = ["_calibrated_model_by_split", "_calibration"]

    def __init__(
        self,
        pipeline: ClassifierPipelineDF[_T_ClassifierDF],
        cv: BaseCrossValidator,
        sample: Sample,
        calibration: str,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            pipeline=pipeline,
            cv=cv,
            sample=sample,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )
        self._calibration = calibration
        self._calibrated_model_by_split: Optional[
            List[ClassifierPipelineDF[CalibratedClassifierCVDF]]
        ] = None

    @staticmethod
    def from_uncalibrated(
        uncalibrated_fit_predict: ClassifierPredictionCV[_T_ClassifierDF],
        calibration: str,
    ) -> "CalibratedClassifierPredictionCV[_T_ClassifierDF]":
        fit_predict = CalibratedClassifierPredictionCV(
            pipeline=uncalibrated_fit_predict._pipeline,
            cv=uncalibrated_fit_predict._cv,
            sample=uncalibrated_fit_predict._sample,
            calibration=calibration,
            n_jobs=uncalibrated_fit_predict._n_jobs,
            shared_memory=uncalibrated_fit_predict._shared_memory,
            verbose=uncalibrated_fit_predict._verbose,
        )
        # if the uncalibrated fit/predict is already fitted, preserve the fit to avoid
        # re-calculating it
        fit_predict._model_by_split = uncalibrated_fit_predict._model_by_split
        return fit_predict

    def _fit(self) -> None:
        super()._fit()

        if self._calibrated_model_by_split is not None:
            return

        sample = self.sample

        self._calibrated_model_by_split: List[
            ClassifierPipelineDF[CalibratedClassifierCVDF]
        ] = self._parallel()(
            delayed(self._calibrate_probabilities_for_split)(
                # we specifically do not clone here, since
                # CalibratedClassifierCV does expect a fitted classifier and
                # clones it by itself
                self._model_by_split[idx],
                sample.select_observations_by_position(positions=test_indices),
                self._calibration,
            )
            for idx, (_, test_indices) in enumerate(
                self.cv.split(sample.features, sample.target)
            )
        )

    def calibrated_model(
        self, split_id: int
    ) -> ClassifierPipelineDF[CalibratedClassifierCVDF]:
        """
        :param split_id: start index of test split
        :return: the pipeline fitted & calibrated for the train split at the given index
        """
        self._fit()
        return self._calibrated_model_by_split[split_id]

    def calibrated_models(
        self
    ) -> Iterator[ClassifierPipelineDF[CalibratedClassifierCVDF]]:
        """
        :return: an iterator of all predictions fitted & calibrated over all train splits
        """
        self._fit()
        return iter(self._calibrated_model_by_split)

    def _pipeline_for_split(
        self, split_id
    ) -> ClassifierPipelineDF[CalibratedClassifierCVDF]:
        return self.calibrated_model(split_id=split_id)

    @staticmethod
    def _calibrate_probabilities_for_split(
        model: ClassifierPipelineDF, test_sample: Sample, calibration: str
    ) -> ClassifierPipelineDF[CalibratedClassifierCVDF]:
        cv = CalibratedClassifierCVDF(
            base_estimator=model.classifier, method=calibration, cv="prefit"
        )

        # clone the pipeline to create a calibrated fit for the current split
        model_calibrated = ClassifierPipelineDF(
            classifier=cv, preprocessing=model.preprocessing
        )

        model_calibrated.fit(X=test_sample.features, y=test_sample.target)

        model_calibrated.predictor_ = ClassifierWrapperDF.from_fitted(
            cv.calibrated_classifiers_[0], cv.features_in
        )

        return model_calibrated
