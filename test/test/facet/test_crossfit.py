import logging

import pytest
from sklearn.model_selection import GridSearchCV

from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression import RandomForestRegressorDF

from ..conftest import check_ranking
from facet.selection import LearnerRanker, MultiClassifierParameterSpace, ParameterSpace
from facet.validation import StratifiedBootstrapCV

log = logging.getLogger(__name__)


def test_prediction_classifier(
    iris_sample_multi_class, cv_stratified_bootstrap: StratifiedBootstrapCV, n_jobs: int
) -> None:

    expected_learner_scores = [0.889, 0.886, 0.885, 0.879]

    # define parameters and crossfit
    ps1 = ParameterSpace(
        ClassifierPipelineDF(classifier=RandomForestClassifierDF(random_state=42))
    )
    ps1.classifier.min_samples_leaf = [16, 32]
    ps1.classifier.n_estimators = [50, 80]

    ps2 = ParameterSpace(
        RegressorPipelineDF(regressor=RandomForestRegressorDF(random_state=42))
    )
    ps2.regressor.min_samples_leaf = [16, 32]
    ps2.regressor.n_estimators = [50, 80]

    with pytest.raises(
        TypeError,
        match="^all candidate estimators must be instances of "
        "ClassifierPipelineDF, but candidate estimators include: "
        "RegressorPipelineDF$",
    ):
        # define an illegal grid list, mixing classification with regression
        MultiClassifierParameterSpace(ps1, ps2)

    model_ranker: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV
    ] = LearnerRanker(
        searcher_factory=GridSearchCV,
        parameter_space=ps1,
        cv=cv_stratified_bootstrap,
        scoring="f1_macro",
        n_jobs=n_jobs,
    )

    with pytest.raises(
        ValueError,
        match="arg sample_weight is not supported, " "use ag sample.weight instead",
    ):
        model_ranker.fit(
            sample=iris_sample_multi_class, sample_weight=iris_sample_multi_class.weight
        )

    model_ranker.fit(sample=iris_sample_multi_class)

    ranking = model_ranker.summary_report()

    log.debug(f"\n{ranking}")

    check_ranking(
        ranking=ranking,
        is_classifier=True,
        expected_scores=expected_learner_scores,
        expected_parameters={
            2: dict(min_samples_leaf=32, n_estimators=50),
            3: dict(min_samples_leaf=32, n_estimators=80),
        },
    )
