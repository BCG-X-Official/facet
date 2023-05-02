"""
Tests for module facet.selection
"""
import logging
from typing import Any, List, Mapping

import numpy as np
import pandas as pd
import pytest
from scipy.stats import randint, reciprocal
from sklearn import datasets
from sklearn.model_selection import GridSearchCV

from pytools.expression import Expression, freeze
from pytools.expression.atomic import Id
from sklearndf import TransformerDF
from sklearndf.classification import SVCDF, RandomForestClassifierDF
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
    LearnerSelector,
    MultiEstimatorParameterSpace,
    ParameterSpace,
)
from facet.validation import BootstrapCV, StratifiedBootstrapCV

log = logging.getLogger(__name__)


def test_learner_selector(
    regressor_parameters: List[ParameterSpace[RegressorPipelineDF[LGBMRegressorDF]]],
    sample: Sample,
    n_jobs: int,
) -> None:

    expected_scores = [
        0.669,
        0.649,
        0.493,
        0.477,
        0.464,
        0.451,
        0.448,
        0.448,
        0.448,
        0.448,
    ]
    expected_learners: List[str] = [
        cls.__name__
        for cls in (
            LinearRegressionDF,
            LinearRegressionDF,
            RandomForestRegressorDF,
            RandomForestRegressorDF,
            AdaBoostRegressorDF,
            AdaBoostRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
        )
    ]
    expected_parameters: Mapping[int, Mapping[str, Any]] = {
        0: dict(fit_intercept=True),
        1: dict(fit_intercept=False),
        3: dict(n_estimators=80),
        4: dict(n_estimators=50),
    }

    # define the circular cross validator with just 5 splits (to speed up testing)
    cv = BootstrapCV(n_splits=5, random_state=42)

    with pytest.raises(
        TypeError,
        match=(
            r"^arg parameter_space requires instances of one of "
            r"{ParameterSpace, MultiEstimatorParameterSpace} but got: int$"
        ),
    ):
        LearnerSelector(
            searcher_type=GridSearchCV,
            parameter_space=1,  # type: ignore
            cv=cv,
        )

    with pytest.raises(
        TypeError,
        match=(
            r"^arg parameter_space requires instances of one of "
            r"{ParameterSpace, MultiEstimatorParameterSpace} but got: int$"
        ),
    ):
        LearnerSelector(
            searcher_type=GridSearchCV,
            parameter_space=[1],  # type: ignore
            cv=cv,
        )

    with pytest.raises(
        TypeError,
        match=(
            r"^arg spaces requires instances of ParameterSpace but got: "
            r"MultiEstimatorParameterSpace$"
        ),
    ):
        multi_ps = MultiEstimatorParameterSpace(*regressor_parameters)
        LearnerSelector(
            searcher_type=GridSearchCV,
            parameter_space=[multi_ps, multi_ps],  # type: ignore
            cv=cv,
        )

    # define the learner selector
    ranker: LearnerSelector[
        RegressorPipelineDF[LGBMRegressorDF], GridSearchCV
    ] = LearnerSelector(
        searcher_type=GridSearchCV,
        parameter_space=regressor_parameters,
        cv=cv,
        scoring="r2",
        n_jobs=n_jobs,
        error_score="raise",
    ).fit(
        sample=sample
    )

    log.debug(f"\n{ranker.summary_report()}")

    assert isinstance(ranker.best_estimator_, RegressorPipelineDF)

    ranking = ranker.summary_report()
    ranking_score = ranking[("score", "test", "mean")]

    assert len(ranking) > 0
    assert all(
        ranking_hi >= ranking_lo
        for ranking_hi, ranking_lo in zip(ranking_score, ranking_score[1:])
    )

    check_ranking(
        ranking=ranking,
        is_classifier=False,
        scores_expected=expected_scores,
        params_expected=expected_parameters,
        candidate_names_expected=expected_learners,
    )


def test_model_selector_no_preprocessing(n_jobs: int) -> None:
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

    model_selector: LearnerSelector[
        ClassifierPipelineDF[SVCDF], GridSearchCV
    ] = LearnerSelector(
        searcher_type=GridSearchCV,
        parameter_space=parameter_space,
        cv=cv,
        n_jobs=n_jobs,
    ).fit(
        sample=test_sample
    )

    summary_report = model_selector.summary_report()
    log.debug(f"\n{summary_report}")

    check_ranking(
        ranking=summary_report,
        is_classifier=True,
        scores_expected=expected_learner_scores,
        params_expected={
            0: dict(C=10, kernel="linear"),
            3: dict(C=1, kernel="rbf"),
        },
    )

    min_best_performance = 0.8
    assert (
        summary_report[("score", "test", "mean")].iloc[0] >= min_best_performance
    ), f"expected the best performance to be at least {min_best_performance}"


def test_parameter_space(simple_preprocessor: TransformerDF) -> None:
    # distributions

    randint_3_10 = randint(3, 10)
    randint_1_32 = randint(1, 32)
    reciprocal_0_01_0_10 = reciprocal(0.01, 0.1)
    reciprocal_0_05_0_10 = reciprocal(0.05, 0.1)

    # parameter space 1

    pipeline_1 = RegressorPipelineDF(
        regressor=RandomForestRegressorDF(random_state=42),
        preprocessing=simple_preprocessor,
    )
    ps_1_name = "rf_regressor"
    ps_1: ParameterSpace[RegressorPipelineDF[RandomForestRegressorDF]] = ParameterSpace(
        pipeline_1, name=ps_1_name
    )
    ps_1.regressor.min_weight_fraction_leaf = reciprocal_0_01_0_10
    ps_1.regressor.max_depth = randint_3_10
    ps_1.regressor.min_samples_leaf = reciprocal_0_05_0_10

    with pytest.raises(
        AttributeError,
        match=r"^unknown parameter name for RandomForestRegressorDF: unknown$",
    ):
        ps_1.regressor.unknown = 1

    # noinspection GrazieInspection
    with pytest.raises(
        TypeError,
        match=(
            r"^expected list or distribution for parameter min_samples_leaf but got: 1$"
        ),
    ):
        ps_1.regressor.min_samples_leaf = 1

    # parameter space 2

    pipeline_2 = RegressorPipelineDF(
        regressor=LGBMRegressorDF(random_state=42),
        preprocessing=simple_preprocessor,
    )
    ps_2_name = "lgbm"
    ps_2: ParameterSpace[RegressorPipelineDF[LGBMRegressorDF]] = ParameterSpace(
        pipeline_2, name=ps_2_name
    )
    ps_2.regressor.max_depth = randint_3_10
    ps_2.regressor.min_child_samples = randint_1_32

    # multi parameter space

    with pytest.raises(
        TypeError,
        match=(
            r"^all parameter spaces must use the same estimator type, "
            r"but got multiple types: classifier, regressor$"
        ),
    ):
        MultiEstimatorParameterSpace(
            ps_1, ps_2, ParameterSpace(ClassifierPipelineDF(classifier=SVCDF()))
        )

    mps = MultiEstimatorParameterSpace(ps_1, ps_2)

    # test

    def regressor_repr(model: Id) -> Expression:
        return Id.RegressorPipelineDF(
            preprocessing=Id.ColumnTransformerDF(
                transformers=[
                    (
                        "impute",
                        Id.SimpleImputerDF(strategy="median"),
                        [
                            "MedInc",
                            "HouseAge",
                            "AveRooms",
                            "AveBedrms",
                            "Population",
                            "AveOccup",
                            "Latitude",
                            "Longitude",
                        ],
                    )
                ]
            ),
            regressor=model(random_state=42),
        )

    assert freeze(mps.to_expression()) == freeze(
        Id.MultiEstimatorParameterSpace(
            Id.ParameterSpace(
                regressor_repr(Id.RandomForestRegressorDF),
                **{
                    "regressor.min_weight_fraction_leaf": (Id.reciprocal(0.01, 0.1)),
                    "regressor.max_depth": Id.randint(3, 10),
                    "regressor.min_samples_leaf": (Id.reciprocal(0.05, 0.1)),
                },
            ),
            Id.ParameterSpace(
                regressor_repr(Id.LGBMRegressorDF),
                **{
                    "regressor.max_depth": Id.randint(3, 10),
                    "regressor.min_child_samples": Id.randint(1, 32),
                },
            ),
        )
    )

    assert mps.estimator.candidate is None

    assert mps.parameters == [
        {
            "candidate": [pipeline_1],
            "candidate_name": [ps_1_name],
            "candidate__regressor__max_depth": randint_3_10,
            "candidate__regressor__min_samples_leaf": reciprocal_0_05_0_10,
            "candidate__regressor__min_weight_fraction_leaf": reciprocal_0_01_0_10,
        },
        {
            "candidate": [pipeline_2],
            "candidate_name": [ps_2_name],
            "candidate__regressor__max_depth": randint_3_10,
            "candidate__regressor__min_child_samples": randint_1_32,
        },
    ]

    assert mps.get_parameters("my_prefix") == [
        {
            "my_prefix__candidate": [pipeline_1],
            "my_prefix__candidate_name": [ps_1_name],
            "my_prefix__candidate__regressor__max_depth": randint_3_10,
            "my_prefix__candidate__regressor__min_samples_leaf": reciprocal_0_05_0_10,
            (
                "my_prefix__candidate__regressor__min_weight_fraction_leaf"
            ): reciprocal_0_01_0_10,
        },
        {
            "my_prefix__candidate": [pipeline_2],
            "my_prefix__candidate_name": [ps_2_name],
            "my_prefix__candidate__regressor__max_depth": randint_3_10,
            "my_prefix__candidate__regressor__min_child_samples": randint_1_32,
        },
    ]


def test_model_selector_regression(
    regressor_parameters: List[ParameterSpace[RegressorPipelineDF[LGBMRegressorDF]]],
    sample: Sample,
    n_jobs: int,
) -> None:
    # define the circular cross validator with just 5 splits (to speed up testing)
    cv = BootstrapCV(n_splits=5, random_state=42)

    with pytest.raises(
        ValueError,
        match=(
            "arg searcher_params must not include the first two positional arguments "
            "of arg searcher_type, but included: param_grid"
        ),
    ):
        LearnerSelector(GridSearchCV, regressor_parameters, param_grid=None)

    ranker: LearnerSelector[
        RegressorPipelineDF[LGBMRegressorDF], GridSearchCV
    ] = LearnerSelector(
        GridSearchCV,
        regressor_parameters,
        scoring="r2",
        cv=cv,
        n_jobs=n_jobs,
    ).fit(
        sample=sample
    )

    assert isinstance(ranker.best_estimator_, RegressorPipelineDF)

    report_df = ranker.summary_report()
    log.debug(report_df.columns.tolist())
    log.debug(f"\n{report_df}")

    assert len(report_df) > 0
    assert isinstance(report_df, pd.DataFrame)

    scores_sr: pd.Series = report_df.loc[:, ("score", "test", "mean")]
    assert all(
        score_hi >= score_lo for score_hi, score_lo in zip(scores_sr, scores_sr[1:])
    )


def test_model_selector_classification(
    iris_sample_multi_class: Sample,
    cv_stratified_bootstrap: StratifiedBootstrapCV,
    n_jobs: int,
) -> None:
    expected_learner_scores = [0.965, 0.964, 0.957, 0.956]

    # define parameters
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
            r"^all parameter spaces must use the same estimator type, "
            "but got multiple types: classifier, regressor$"
        ),
    ):
        # define an illegal grid list, mixing classification with regression
        MultiEstimatorParameterSpace(ps1, ps2)

    model_selector: LearnerSelector[
        ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV
    ] = LearnerSelector(
        searcher_type=GridSearchCV,
        parameter_space=ps1,
        cv=cv_stratified_bootstrap,
        scoring="f1_macro",
        n_jobs=n_jobs,
    )

    with pytest.raises(
        ValueError,
        match=(
            "arg sample_weight is not supported, use 'weight' property "
            "of arg sample instead"
        ),
    ):
        model_selector.fit(
            sample=iris_sample_multi_class, sample_weight=iris_sample_multi_class.weight
        )

    model_selector.fit(sample=iris_sample_multi_class)

    ranking = model_selector.summary_report()

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
