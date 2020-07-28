import logging

from gamma.ml import Sample
from gamma.ml.selection import LearnerRanker, ParameterGrid
from gamma.ml.validation import BootstrapCV
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF
from test.gamma.ml import check_ranking, disable_warnings

log = logging.getLogger(__name__)


def test_prediction_classifier(
    iris_sample: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> None:
    disable_warnings()

    expected_learner_scores = [0.889, 0.889, 0.853, 0.815]

    # define parameters and crossfit
    grid = ParameterGrid(
        pipeline=ClassifierPipelineDF(
            classifier=RandomForestClassifierDF(random_state=42), preprocessing=None
        ),
        learner_parameters={"min_samples_leaf": [16, 32], "n_estimators": [50, 80]},
    )

    model_ranker: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ] = LearnerRanker(
        grid=grid, cv=cv_bootstrap, scoring="f1_macro", n_jobs=n_jobs, random_state=42
    ).fit(
        sample=iris_sample
    )

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking(),
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
