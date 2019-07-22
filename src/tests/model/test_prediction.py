import logging
import warnings
from itertools import combinations
from typing import *

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

from yieldengine import Sample
from yieldengine.model import ModelPipelineDF
from yieldengine.model.prediction import ClassifierFitCV, ProbabilityCalibrationMethod
from yieldengine.model.selection import (
    ModelEvaluation,
    ModelGrid,
    ModelRanker,
    summary_report,
)
from yieldengine.sklearndf.classification import RandomForestClassifierDF

log = logging.getLogger(__name__)

K_FOLDS: int = 5
TEST_RATIO = 1 / K_FOLDS
N_SPLITS = K_FOLDS * 2
CALIBRATION_DIFF_THRESHOLD = 0.3


def test_prediction_classifier(available_cpus: int, iris_sample: Sample) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameters and models
    models = [
        ModelGrid(
            pipeline=ModelPipelineDF(
                predictor=RandomForestClassifierDF(), preprocessing=None
            ),
            estimator_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    test_sample: Sample = iris_sample

    model_ranker: ModelRanker = ModelRanker(
        grids=models, cv=test_cv, scoring="f1_macro"
    )

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=available_cpus
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    model_evaluation = model_ranking[0]

    # store proba-results
    proba_results = {}

    # test various ProbabilityCalibrationMethods for a classifier:
    for calibration_method in (
        None,
        ProbabilityCalibrationMethod.ISOTONIC,
        ProbabilityCalibrationMethod.SIGMOID,
    ):

        model_fit = ClassifierFitCV(
            model=model_evaluation.model,
            cv=test_cv,
            sample=test_sample,
            calibration=calibration_method,
            n_jobs=available_cpus,
        )

        # test predictions_for_all_samples
        predictions_df: pd.DataFrame = model_fit.predictions_for_all_splits()
        assert ClassifierFitCV.F_PREDICTION in predictions_df.columns
        assert ClassifierFitCV.F_TARGET in predictions_df.columns

        # check number of split ids
        assert (
            predictions_df.index.get_level_values(
                level=ClassifierFitCV.F_SPLIT_ID
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

        # test probabilities for all samples
        proba_df: pd.DataFrame = model_fit.probabilities_for_all_splits()

        for target_class in test_sample.target.unique():
            assert target_class in proba_df.columns

        # store probabilites for class "setosa"
        proba_results[calibration_method] = proba_df.loc[:, "setosa"]

        # test log-probabilities for uncalibrated classifier:
        if calibration_method is None:
            log_proba_df = model_fit.log_probabilities_for_all_splits()
            assert log_proba_df.shape == proba_df.shape
            assert np.all(
                log_proba_df.loc[:, "setosa"] == np.log(proba_df.loc[:, "setosa"])
            )

    for p1, p2 in combinations(
        [
            ProbabilityCalibrationMethod.ISOTONIC,
            ProbabilityCalibrationMethod.SIGMOID,
            None,
        ],
        2,
    ):
        cumulative_diff = np.sum(np.abs(proba_results[p1] - proba_results[p2]))

        log.info(f"Cumulative diff of calibration {p1} vs {p2} is: {cumulative_diff}")

        assert cumulative_diff > CALIBRATION_DIFF_THRESHOLD
