import logging

import pytest

from facet import Sample
from facet.selection import LearnerGrid, LearnerRanker
from facet.validation import StratifiedBootstrapCV
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF
from . import check_ranking

log = logging.getLogger(__name__)


def test_prediction_classifier(
    iris_sample: Sample, cv_stratified_bootstrap: StratifiedBootstrapCV, n_jobs: int
) -> None:

    expected_learner_scores = [0.889, 0.886, 0.885, 0.879]

    # define parameters and crossfit
    grids = LearnerGrid(
        pipeline=ClassifierPipelineDF(
            classifier=RandomForestClassifierDF(random_state=42), preprocessing=None
        ),
        learner_parameters={"min_samples_leaf": [16, 32], "n_estimators": [50, 80]},
    )

    model_ranker: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ] = LearnerRanker(
        grids=grids,
        cv=cv_stratified_bootstrap,
        scoring="f1_macro",
        n_jobs=n_jobs,
        random_state=42,
    )

    model_ranker.fit(sample=iris_sample)

    with pytest.raises(
        ValueError, match="do not use arg sample_weight to pass sample weights"
    ):
        model_ranker.fit(sample=iris_sample, sample_weight=iris_sample.weight)

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking,
        expected_scores=expected_learner_scores,
        expected_learners=[RandomForestClassifierDF] * 4,
        expected_parameters={
            2: dict(classifier__min_samples_leaf=32, classifier__n_estimators=50),
            3: dict(classifier__min_samples_leaf=32, classifier__n_estimators=80),
        },
    )

    # consider: model_with_type(...) function for ModelRanking
    crossfit = model_ranker.best_model_crossfit

    assert crossfit.is_fitted
