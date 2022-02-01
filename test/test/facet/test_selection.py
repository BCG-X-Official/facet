"""
Tests for module facet.selection
"""

import logging

import numpy as np
import pandas as pd
import pytest
from scipy.stats import loguniform, randint, zipfian
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

from pytools.expression import freeze
from pytools.expression.atomic import Id
from sklearndf import TransformerDF
from sklearndf.classification import SVCDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression import (
    AdaBoostRegressorDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
)
from sklearndf.regression.extra import LGBMRegressorDF

from ..conftest import check_ranking
from facet.data import Sample
from facet.selection import (
    LearnerRanker,
    MultiClassifierParameterSpace,
    MultiRegressorParameterSpace,
    ParameterSpace,
)
from facet.validation import BootstrapCV

log = logging.getLogger(__name__)


def test_model_ranker(
    regressor_parameters: MultiRegressorParameterSpace, sample: Sample, n_jobs: int
) -> None:

    expected_scores = [
        0.840,
        0.837,
        0.812,
        0.812,
        0.793,
        0.790,
        0.758,
        0.758,
        0.758,
        0.758,
    ]
    expected_learners = [
        RandomForestRegressorDF,
        RandomForestRegressorDF,
        LinearRegressionDF,
        LinearRegressionDF,
        AdaBoostRegressorDF,
        AdaBoostRegressorDF,
        LGBMRegressorDF,
        LGBMRegressorDF,
        LGBMRegressorDF,
        LGBMRegressorDF,
    ]
    expected_parameters = {
        0: dict(n_estimators=80),
        1: dict(n_estimators=50),
        4: dict(n_estimators=50),
        5: dict(n_estimators=80),
    }

    # define the circular cross validator with just 5 splits (to speed up testing)
    cv = BootstrapCV(n_splits=5, random_state=42)

    ranker: LearnerRanker[RegressorPipelineDF, GridSearchCV] = LearnerRanker(
        searcher_factory=GridSearchCV,
        parameter_space=regressor_parameters,
        cv=cv,
        scoring="r2",
        n_jobs=n_jobs,
    ).fit(sample=sample)

    log.debug(f"\n{ranker.summary_report()}")

    assert isinstance(ranker.best_estimator_, RegressorPipelineDF)

    ranking = ranker.summary_report()
    ranking_score = ranking["mean_test_score"]

    assert len(ranking) > 0
    assert all(
        ranking_hi >= ranking_lo
        for ranking_hi, ranking_lo in zip(ranking_score, ranking_score[1:])
    )

    check_ranking(
        ranking=ranking,
        is_classifier=False,
        expected_scores=expected_scores,
        expected_parameters=expected_parameters,
        expected_learners=expected_learners,
    )


def test_model_ranker_no_preprocessing(n_jobs) -> None:

    expected_learner_scores = [0.961, 0.957, 0.957, 0.936]

    # define a yield-engine circular CV:
    cv = BootstrapCV(n_splits=5, random_state=42)

    # define parameters and pipeline
    parameter_space = ParameterSpace(
        ClassifierPipelineDF(classifier=SVCDF(gamma="scale"), preprocessing=None)
    )
    parameter_space.classifier.kernel = ["linear", "rbf"]
    parameter_space.classifier.C = [1, 10]

    #  load scikit-learn test-data and convert to pd
    iris = datasets.load_iris()
    test_data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=[*iris["feature_names"], "target"],
    )
    test_sample: Sample = Sample(observations=test_data, target_name="target")

    model_ranker: LearnerRanker[
        ClassifierPipelineDF[SVCDF], GridSearchCV
    ] = LearnerRanker(
        searcher_factory=GridSearchCV,
        parameter_space=parameter_space,
        cv=cv,
        n_jobs=n_jobs,
    ).fit(
        sample=test_sample
    )

    summary_report = model_ranker.summary_report()
    log.debug(f"\n{summary_report}")

    check_ranking(
        ranking=summary_report,
        is_classifier=True,
        expected_scores=expected_learner_scores,
        expected_parameters={
            0: dict(C=10, kernel="linear"),
            3: dict(C=1, kernel="rbf"),
        },
    )

    assert (
        summary_report["mean_test_score"].iloc[0] >= 0.8
    ), "expected a best performance of at least 0.8"


def test_parameter_space(
    sample: Sample, simple_preprocessor: TransformerDF, n_jobs: int
) -> None:

    # distributions

    randint_3_10 = randint(3, 10)
    loguniform_0_01_0_10 = loguniform(0.01, 0.1)
    loguniform_0_05_0_10 = loguniform(0.05, 0.1)
    zipfian_1_32 = zipfian(1.0, 32)

    # parameter space 1

    pipeline_1 = RegressorPipelineDF(
        regressor=RandomForestRegressorDF(random_state=42),
        preprocessing=simple_preprocessor,
    )
    ps_1_name = "rf_regressor"
    ps_1 = ParameterSpace(pipeline_1, name=ps_1_name)
    ps_1.regressor.min_weight_fraction_leaf = loguniform_0_01_0_10
    ps_1.regressor.max_depth = randint_3_10
    ps_1.regressor.min_samples_leaf = loguniform_0_05_0_10

    with pytest.raises(
        AttributeError,
        match=r"^unknown parameter name for RandomForestRegressorDF: unknown$",
    ):
        ps_1.regressor.unknown = 1

    with pytest.raises(
        TypeError,
        match=(
            "^expected list or distribution for parameter min_samples_leaf "
            "but got: 1$"
        ),
    ):
        ps_1.regressor.min_samples_leaf = 1

    # parameter space 2

    pipeline_2 = RegressorPipelineDF(
        regressor=LGBMRegressorDF(random_state=42),
        preprocessing=simple_preprocessor,
    )
    ps_2_name = "lgbm"
    ps_2 = ParameterSpace(pipeline_2, name=ps_2_name)
    ps_2.regressor.max_depth = randint_3_10
    ps_2.regressor.min_child_samples = zipfian_1_32

    # multi parameter space

    with pytest.raises(
        TypeError,
        match=(
            r"^arg estimator_type must be a subclass of ClassifierPipelineDF but is: "
            r"RegressorPipelineDF$"
        ),
    ):
        # noinspection PyTypeChecker
        MultiClassifierParameterSpace(ps_1, ps_2, estimator_type=RegressorPipelineDF)

    with pytest.raises(
        TypeError,
        match=(
            r"^all candidate estimators must be instances of ClassifierPipelineDF, "
            r"but candidate estimators include: RegressorPipelineDF$"
        ),
    ):
        # noinspection PyTypeChecker
        MultiClassifierParameterSpace(ps_1, ps_2)

    mps = MultiRegressorParameterSpace(ps_1, ps_2)

    # test

    assert freeze(mps.to_expression()) == freeze(
        Id.MultiRegressorParameterSpace(
            None,
            [
                Id.ParameterSpace(
                    candidate=pipeline_1.to_expression(),
                    **{
                        "candidate.regressor.min_weight_fraction_leaf": (
                            Id.loguniform(0.01, 0.1)
                        ),
                        "candidate.regressor.max_depth": Id.randint(3, 10),
                        "candidate.regressor.min_samples_leaf": (
                            Id.loguniform(0.05, 0.1)
                        ),
                    },
                ),
                Id.ParameterSpace(
                    candidate=pipeline_2.to_expression(),
                    **{
                        "candidate.regressor.max_depth": Id.randint(3, 10),
                        "candidate.regressor.min_child_samples": Id.zipfian(1.0, 32),
                    },
                ),
            ],
        )
    )

    assert mps.estimator.candidate is None

    assert mps.parameters == [
        {
            "candidate": [pipeline_1],
            "candidate_name": [ps_1_name],
            "candidate__regressor__max_depth": randint_3_10,
            "candidate__regressor__min_samples_leaf": loguniform_0_05_0_10,
            "candidate__regressor__min_weight_fraction_leaf": loguniform_0_01_0_10,
        },
        {
            "candidate": [pipeline_2],
            "candidate_name": [ps_2_name],
            "candidate__regressor__max_depth": randint_3_10,
            "candidate__regressor__min_child_samples": zipfian_1_32,
        },
    ]


def test_learner_ranker(
    regressor_parameters: MultiRegressorParameterSpace, sample: Sample, n_jobs: int
) -> None:

    # define the circular cross validator with just 5 splits (to speed up testing)
    cv = BootstrapCV(n_splits=5, random_state=42)

    with pytest.raises(
        ValueError,
        match=(
            "arg searcher_params must not include the first two positional arguments "
            "of arg searcher_factory, but included: param_grid"
        ),
    ):
        LearnerRanker(GridSearchCV, regressor_parameters, param_grid=None)

    ranker: LearnerRanker[RegressorPipelineDF, GridSearchCV] = LearnerRanker(
        GridSearchCV,
        regressor_parameters,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
    ).fit(sample=sample)

    assert isinstance(ranker.best_estimator_, RegressorPipelineDF)

    report_df = ranker.summary_report()
    log.debug(report_df.columns.tolist())
    log.debug(f"\n{report_df}")

    assert len(report_df) > 0
    assert isinstance(report_df, pd.DataFrame)

    scores_sr: pd.Series = report_df.loc[:, "mean_test_score"]
    assert all(
        score_hi >= score_lo for score_hi, score_lo in zip(scores_sr, scores_sr[1:])
    )
