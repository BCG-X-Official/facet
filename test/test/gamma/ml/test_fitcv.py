import logging
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

from gamma.ml import Sample
from gamma.ml.predictioncv import (
    ClassifierPredictionCV,
    CalibratedClassifierPredictionCV,
)
from gamma.ml.selection import ParameterGrid, ClassifierRanker
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF

log = logging.getLogger(__name__)

K_FOLDS: int = 5
TEST_RATIO = 1 / K_FOLDS
N_SPLITS = K_FOLDS * 2
CALIBRATION_DIFF_THRESHOLD = 0.3

CHKSUM_CLASSIFIER_PREDICTIONS = 13680176299872221154


def test_prediction_classifier(n_jobs, iris_sample: Sample) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameters and predictions
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(), preprocessing=None
            ),
            learner_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    test_sample: Sample = iris_sample

    model_ranker: ClassifierRanker = ClassifierRanker(
        grid=models, sample=test_sample, cv=test_cv, scoring="f1_macro", n_jobs=n_jobs
    )

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    # consider: model_with_type(...) function for ModelRanking
    fit_predict = model_ranker.best_model_predictions

    # store proba-results
    proba_results = {}

    # test various ProbabilityCalibrationMethods for a classifier:
    for calibration_method in (
        None,
        ClassifierPredictionCV.CALIBRATION_ISOTONIC,
        ClassifierPredictionCV.CALIBRATION_SIGMOID,
    ):

        if calibration_method is None:
            calibrated = fit_predict
        else:
            calibrated = CalibratedClassifierPredictionCV.from_uncalibrated(
                uncalibrated_fit_predict=fit_predict, calibration=calibration_method
            )

        # test predictions_for_all_samples
        predictions_df = calibrated.predictions_for_all_splits()
        assert ClassifierPredictionCV.COL_PREDICTION in predictions_df.columns
        assert ClassifierPredictionCV.COL_TARGET in predictions_df.columns

        # check number of split ids
        assert (
            predictions_df.index.get_level_values(
                level=ClassifierPredictionCV.COL_SPLIT_ID
            ).nunique()
            == N_SPLITS
        )

        # check correct number of rows
        allowed_variance = 0.01
        assert (
            (len(test_sample) * (TEST_RATIO - allowed_variance) * N_SPLITS)
            <= len(predictions_df)
            <= (len(test_sample) * (TEST_RATIO + allowed_variance) * N_SPLITS)
        )

        assert (
            np.sum(hash_pandas_object(predictions_df).values)
            == CHKSUM_CLASSIFIER_PREDICTIONS
        )

        # test probabilities for all samples
        proba_df: pd.DataFrame = calibrated.probabilities_for_all_splits()

        for target_class in test_sample.target.unique():
            assert target_class in proba_df.columns

        # store probabilites for class "setosa"
        proba_results[calibration_method] = proba_df.loc[:, "setosa"]

        # test log-probabilities for uncalibrated classifier:
        if calibration_method is None:
            log_proba_df = calibrated.log_probabilities_for_all_splits()
            assert log_proba_df.shape == proba_df.shape
            assert np.all(
                log_proba_df.loc[:, "setosa"] == np.log(proba_df.loc[:, "setosa"])
            )

    for p1, p2 in combinations(
        [
            ClassifierPredictionCV.CALIBRATION_ISOTONIC,
            ClassifierPredictionCV.CALIBRATION_SIGMOID,
            None,
        ],
        2,
    ):
        cumulative_diff = np.sum(np.abs(proba_results[p1] - proba_results[p2]))

        log.info(f"Cumulative diff of calibration {p1} vs {p2} is: {cumulative_diff}")

        assert cumulative_diff > CALIBRATION_DIFF_THRESHOLD
