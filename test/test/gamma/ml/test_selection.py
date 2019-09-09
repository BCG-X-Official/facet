import hashlib
import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
from sklearn import datasets

from gamma.ml import Sample
from gamma.ml.selection import (
    ModelEvaluation,
    ModelParameterGrid,
    ModelRanker,
    summary_report,
)
from gamma.ml.validation import CircularCrossValidator
from gamma.sklearndf.classification import SVCDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF

log = logging.getLogger(__name__)

CHKSUM_SUMMARY_REPORT = "925b6623fa1b10bee69cb179b03a6c52"


def test_model_ranker(
    batch_table: pd.DataFrame, regressor_grids, sample: Sample, n_jobs
) -> None:
    # define the circular cross validator with just 5 splits (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_splits=5)

    model_ranker: ModelRanker = ModelRanker(
        grids=regressor_grids, cv=circular_cv, scoring="r2"
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        sample=sample, n_jobs=n_jobs
    )

    assert len(model_ranking) > 0
    assert isinstance(model_ranking[0], ModelEvaluation)
    assert (
        model_ranking[0].ranking_score
        >= model_ranking[1].ranking_score
        >= model_ranking[2].ranking_score
        >= model_ranking[3].ranking_score
        >= model_ranking[4].ranking_score
        >= model_ranking[-1].ranking_score
    )

    # check if parameters set for estimators actually match expected:
    for scoring in model_ranking:
        assert set(scoring.model.get_params()).issubset(scoring.model.get_params())

    log.debug(f"\n{model_ranking}")
    assert CHKSUM_SUMMARY_REPORT == (
        hashlib.md5(summary_report(model_ranking).encode("utf-8")).hexdigest()
    )


def test_model_ranker_no_preprocessing(n_jobs) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a yield-engine circular CV:
    cv = CircularCrossValidator(test_ratio=0.21, num_splits=50)

    # define parameters and model
    models = [
        ModelParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=SVCDF(gamma="scale"), preprocessing=None
            ),
            estimator_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        )
    ]

    model_ranker: ModelRanker = ModelRanker(grids=models, cv=cv)

    #  load sklearn test-data and convert to pd
    iris = datasets.load_iris()
    test_data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    test_sample: Sample = Sample(observations=test_data, target_name="target")

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=n_jobs
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    assert (
        model_ranking[0].ranking_score >= 0.8
    ), "expected a best performance of at least 0.8"
