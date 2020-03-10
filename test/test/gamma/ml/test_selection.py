import logging
import warnings

import numpy as np
import pandas as pd
from sklearn import datasets

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.selection import (
    ClassifierRanker,
    LearnerEvaluation,
    ParameterGrid,
    RegressorRanker,
)
from gamma.ml.validation import BootstrapCV
from gamma.sklearndf.classification import SVCDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF
from test.gamma.ml import check_ranking

log = logging.getLogger(__name__)


def test_model_ranker(
    batch_table: pd.DataFrame, regressor_grids, sample: Sample, n_jobs
) -> None:

    checksum_learner_scores = 4.994168961240268
    checksum_learner_ranks = "97a4b0f59f52daab7b6d223075267548"

    # define the circular cross validator with just 5 splits (to speed up testing)
    cv = BootstrapCV(n_splits=5, random_state=42)

    ranker = RegressorRanker(
        grid=regressor_grids, cv=cv, scoring="r2", n_jobs=n_jobs
    ).fit(sample=sample)
    assert isinstance(ranker.best_model_crossfit, LearnerCrossfit)

    ranking = ranker.ranking()
    assert len(ranking) > 0
    assert isinstance(ranking[0], LearnerEvaluation)
    assert (
        ranking[0].ranking_score
        >= ranking[1].ranking_score
        >= ranking[2].ranking_score
        >= ranking[3].ranking_score
        >= ranking[4].ranking_score
        >= ranking[-1].ranking_score
    )

    # check if parameters set for estimators actually match expected:
    for validation in ranker.ranking():
        assert set(validation.pipeline.get_params()).issubset(
            validation.pipeline.get_params()
        )

    check_ranking(
        ranking=ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )


def test_model_ranker_no_preprocessing(n_jobs) -> None:
    checksum_learner_scores = 3.6520651615505493
    checksum_learner_ranks = "616a57e7656951a9109bf425be50ee34"

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a yield-engine circular CV:
    cv = BootstrapCV(n_splits=5, random_state=42)

    # define parameters and pipeline
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=SVCDF(gamma="scale"), preprocessing=None
            ),
            learner_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        )
    ]

    #  load scikit-learn test-data and convert to pd
    iris = datasets.load_iris()
    test_data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    test_sample: Sample = Sample(observations=test_data, target="target")

    model_ranker: ClassifierRanker = ClassifierRanker(
        grid=models, cv=cv, n_jobs=n_jobs
    ).fit(sample=test_sample)

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    assert (
        model_ranker.ranking()[0].ranking_score >= 0.8
    ), "expected a best performance of at least 0.8"
