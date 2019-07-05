import copy
import logging
from typing import *

import pandas as pd
from joblib import delayed, Parallel
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from yieldengine import Sample
from yieldengine.model import Model

log = logging.getLogger(__name__)


class PredictorCV:
    __slots__ = [
        "_model",
        "_cv",
        "_sample",
        "_predictions_for_all_samples",
        "_model_by_split",
        "_n_jobs",
        "_verbose",
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
        verbose: int = 0,
    ) -> None:
        self._model = model
        self._cv = cv
        self._sample = sample
        self._model_by_split: Optional[Dict[int, Model]] = None
        self._predictions_for_all_samples: Optional[pd.DataFrame] = None
        self._n_jobs = n_jobs
        self._verbose = verbose

    @property
    def cv(self) -> BaseCrossValidator:
        return self._cv

    @property
    def sample(self) -> Sample:
        return self._sample

    @property
    def model_by_split(self) -> Optional[Dict[int, Model]]:
        return self._model_by_split

    @property
    def split_ids(self) -> Optional[Set[int]]:
        return set() if self.model_by_split is None else self.model_by_split.keys()

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    def model(self, split_id: int) -> Model:
        """
        :param split_id: start index of test split
        :return: the model that was used to predict the dependent variable of
        the test split
        """
        if self._model_by_split is None:
            self.predictions_for_all_samples()
        return self._model_by_split[split_id]

    def predictions_for_split(self, split_id: int) -> pd.Series:
        all_predictions = self.predictions_for_all_samples()
        return all_predictions.loc[
            all_predictions[PredictorCV.F_SPLIT_ID] == split_id,
            PredictorCV.F_PREDICTION,
        ]

    def estimator(self, split: int) -> BaseEstimator:
        """
        :param split: start index of test split
        :return: the estimator that was used to predict the dependent variable of
        the test split
        """
        if self._model_by_split is None:
            self.predictions_for_all_samples()
        return self._model_by_split[split].estimator

    def _is_fitted(self) -> bool:
        return self._model_by_split is not None

    def fit(self) -> None:
        if self._is_fitted():
            return

        self._model_by_split: Dict[int, Model] = {}

        sample = self.sample
        args = []
        split_ids = []

        for split_id, (train_indices, _) in enumerate(
            self.cv.split(sample.features, sample.target)
        ):
            train_sample = sample.select_observations(numbers=train_indices)
            model = self._model.clone()
            args.append((model, train_sample))
            split_ids.append(split_id)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self._verbose)

        models = parallel(
            delayed(_fit_model_for_split)(model, train_sample)
            for model, train_sample in args
        )

        for split_id, model in zip(split_ids, models):
            self._model_by_split[split_id] = model
        log.info("Finished to fit PredictorCV.")

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

        if self._predictions_for_all_samples is None:

            if not self._is_fitted():
                self.fit()

            sample = self.sample

            def _predictions_for_all_samples() -> pd.DataFrame:
                parallel = Parallel(n_jobs=self.n_jobs, verbose=self._verbose)

                predictions_per_split: Iterable[pd.DataFrame] = parallel(
                    delayed(_predictions_for_split)(
                        split_id=split_id,
                        test_sample=sample.select_observations(numbers=test_indices),
                        test_model=self.model(split_id=split_id),
                    )
                    for split_id, (_, test_indices) in enumerate(
                        self.cv.split(sample.features, sample.target)
                    )
                )

                predictions: pd.DataFrame = pd.concat(predictions_per_split).join(
                    sample.target.rename(PredictorCV.F_TARGET)
                )

                return predictions

            self._predictions_for_all_samples = _predictions_for_all_samples()

        return self._predictions_for_all_samples

    def copy_with_sample(self, sample: Sample):
        copied_predictor = copy.copy(self)
        copied_predictor._sample = sample
        copied_predictor._predictions_for_all_samples = None
        return copied_predictor

#
# we move all parallelisable code outside of the PredictorCV class as this brings a
# major performance benefit under Windows
#

def _predictions_for_split(
    split_id: int, test_sample: Sample, test_model: Model
) -> pd.DataFrame:
    """
    Compute predictions for a given split.

    :param split_id: the split id
    :param test_sample: the `Sample` of the split test set
    :param test_model: the fitted model for the split
    :return: dataframe with columns `split_id` and `prediction`.
    """
    return pd.DataFrame(
        data={
            PredictorCV.F_SPLIT_ID: split_id,
            PredictorCV.F_PREDICTION: test_model.pipeline.predict(
                X=test_sample.features
            ),
        },
        index=test_sample.index,
    )


def _fit_model_for_split(model: Model, train_sample: Sample):
    """
    Fit a model using a sample.

    :param model:  the `Model` to fit
    :param train_sample: `Sample` to fit on
    :return: the fitted `Model`
    """
    model.pipeline.fit(X=train_sample.features, y=train_sample.target)
    return model
