import logging

import pytest
from sklearn.model_selection import GridSearchCV

from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression import RandomForestRegressorDF

from ..conftest import check_ranking
from facet.selection import LearnerRanker, MultiEstimatorParameterSpace, ParameterSpace
from facet.validation import StratifiedBootstrapCV

log = logging.getLogger(__name__)


def test_prediction_classifier(
    iris_sample_multi_class, cv_stratified_bootstrap: StratifiedBootstrapCV, n_jobs: int
) -> None:

    expected_learner_scores = [0.965, 0.964, 0.957, 0.956]

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
        match=(
            "^all candidate estimators must have the same estimator type, "
            "but got multiple types: classifier, regressor$"
        ),
    ):
        # define an illegal grid list, mixing classification with regression
        MultiEstimatorParameterSpace(ps1, ps2)

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
        match="arg sample_weight is not supported, use arg sample.weight instead",
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
        scores_expected=expected_learner_scores,
        params_expected={
            2: dict(min_samples_leaf=32, n_estimators=50),
            3: dict(min_samples_leaf=32, n_estimators=80),
        },
    )
