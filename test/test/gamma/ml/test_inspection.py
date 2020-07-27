"""
Model inspector tests.
"""
import functools
import logging
import warnings
from distutils import version
from typing import *

import numpy as np
import pandas as pd
import pytest
import shap
from pandas.core.util.hashing import hash_pandas_object
from pandas.util.testing import assert_frame_equal
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import ClassifierInspector, RegressorInspector
from gamma.ml.selection import LearnerRanker, ParameterGrid
from gamma.ml.validation import BootstrapCV
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from test.gamma.ml import check_ranking

# noinspection PyMissingOrEmptyDocstring

log = logging.getLogger(__name__)


@pytest.fixture
def iris_classifier_ranker(
    cv_bootstrap: BootstrapCV, iris_sample_binary: Sample, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    # define parameters and crossfit
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(random_state=42), preprocessing=None
            ),
            learner_parameters={"n_estimators": [10, 50], "min_samples_leaf": [16, 32]},
        )
    ]
    # pipeline inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    return LearnerRanker(
        grid=models,
        cv=cv_bootstrap,
        scoring="f1_macro",
        # shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=iris_sample_binary)


@pytest.fixture
def iris_classifier_ranker_dual_target(
    cv_bootstrap: BootstrapCV, iris_sample_binary_dual_target: Sample, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    # define parameters and crossfit
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(random_state=42), preprocessing=None
            ),
            learner_parameters={"n_estimators": [10, 50], "min_samples_leaf": [16, 32]},
        )
    ]
    # pipeline inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    return LearnerRanker(
        grid=models,
        cv=cv_bootstrap,
        scoring="f1_macro",
        # shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=iris_sample_binary_dual_target)


@pytest.fixture
def iris_classifier_crossfit(
    iris_classifier_ranker: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ]
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker.best_model_crossfit


def test_model_inspection(
    regressor_grids: Sequence[ParameterGrid[RegressorPipelineDF]],
    regressor_ranker: LearnerRanker[RegressorPipelineDF],
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF],
    feature_names: Set[str],
    regressor_inspector: RegressorInspector,
    cv_kfold,
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs: int,
    fast_execution: bool,
) -> None:
    # define checksums for this test
    if fast_execution:
        checksum_shap = 7678718855667032507

        expected_scores = [
            0.418,
            0.4,
            0.386,
            0.385,
            0.122,
            0.122,
            -0.074,
            -0.074,
            -0.074,
            -0.074,
        ]
    else:
        checksum_shap = 1956741545033811954

        expected_scores = [
            0.202,
            0.160,
            0.111,
            0.056,
            0.031,
            0.010,
            0.010,
            0.010,
            0.007,
            0.007,
        ]

    log.debug(f"\n{regressor_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=regressor_ranker.ranking(),
        expected_scores=expected_scores,
        expected_learners=None,
        expected_parameters=None,
    )

    # using an invalid consolidation method raises an exception
    with pytest.raises(ValueError, match="unknown consolidation method: invalid"):
        regressor_inspector.shap_values(consolidate="invalid")

    shap_values_raw = regressor_inspector.shap_values(consolidate=None)
    shap_values_mean = regressor_inspector.shap_values()
    shap_values_std = regressor_inspector.shap_values(consolidate="std")

    # method shap_values without parameter is equal to "mean" consolidation
    assert shap_values_mean.equals(regressor_inspector.shap_values(consolidate="mean"))

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values_mean) == len(sample)

    # index names
    assert shap_values_mean.index.names == [Sample.COL_OBSERVATION]
    assert shap_values_mean.columns.names == [Sample.COL_FEATURE]
    assert shap_values_std.index.names == [Sample.COL_OBSERVATION]
    assert shap_values_std.columns.names == [Sample.COL_FEATURE]
    assert shap_values_raw.index.names == [
        RegressorInspector.COL_SPLIT,
        Sample.COL_OBSERVATION,
    ]
    assert shap_values_raw.columns.names == [Sample.COL_FEATURE]

    # column index
    assert set(shap_values_mean.columns) == feature_names

    # check that the SHAP values add up to the predictions
    mean_predictions = shap_values_mean.sum(axis=1)
    shap_totals_raw = shap_values_raw.sum(axis=1)

    for split_id, model in enumerate(best_lgbm_crossfit.models()):
        # for each model in the crossfit, calculate the difference between total
        # SHAP values and prediction for every observation. This is always the same
        # constant value, so `mad` (mean absolute deviation) is zero

        shap_minus_pred = shap_totals_raw.xs(key=split_id, level=0) - model.predict(
            X=sample.features
        )
        assert (
            round(shap_minus_pred.mad(), 12) == 0.0
        ), f"predictions matching total SHAP for split {split_id}"

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_values_mean.round(decimals=4)).values)
        == checksum_shap
    )

    #  test the ModelInspector with a custom ExplainerFactory:
    def _ef(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:

        try:
            te = TreeExplainer(
                model=estimator, feature_dependence="independent", data=data
            )

            if version.LooseVersion(shap.__version__) >= "0.32":
                log.debug(
                    f"Version of shap is {shap.__version__} - "
                    f"setting check_additivity=False"
                )
                te.shap_values = functools.partial(
                    te.shap_values, check_additivity=False
                )
                return te
            else:
                return te
        except Exception as e:
            log.debug(
                f"failed to instantiate shap.TreeExplainer:{str(e)},"
                "using shap.KernelExplainer as fallback"
            )
            # noinspection PyUnresolvedReferences
            return KernelExplainer(model=estimator.predict, data=data)

    # noinspection PyTypeChecker
    inspector_2 = RegressorInspector(
        explainer_factory=_ef, shap_interaction=False, n_jobs=n_jobs
    ).fit(crossfit=best_lgbm_crossfit)
    inspector_2.shap_values()

    linkage_tree = inspector_2.feature_association_linkage()

    print()
    DendrogramDrawer(style="text").draw(data=linkage_tree, title="Test")


def test_binary_classifier_ranking(
    iris_classifier_ranker: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ]
) -> None:
    expected_learner_scores = [0.858, 0.835, 0.784, 0.689]

    log.debug(f"\n{iris_classifier_ranker.summary_report(max_learners=10)}")
    check_ranking(
        ranking=iris_classifier_ranker.ranking(),
        expected_scores=expected_learner_scores,
        expected_learners=[RandomForestClassifierDF] * 4,
        expected_parameters={
            2: dict(classifier__min_samples_leaf=32, classifier__n_estimators=50),
            3: dict(classifier__min_samples_leaf=32, classifier__n_estimators=10),
        },
    )


def test_model_inspection_classifier(
    iris_sample_binary: Sample,
    iris_classifier_crossfit: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    model_inspector = ClassifierInspector(shap_interaction=False, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit
    )

    # calculate the shap value matrix, without any consolidation
    shap_values = model_inspector.shap_values(consolidate=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit
    )

    shap_matrix_mean = model_inspector.shap_values()

    # is the consolidation correct?
    assert_frame_equal(shap_matrix_mean, shap_values.mean(level=1))

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix_mean) == len(iris_sample_binary)

    # Shap decomposition matrices (feature dependencies)
    _check_feature_dependency_matrices(
        model_inspector=model_inspector,
        feature_names=iris_sample_binary.feature_columns,
        synergy=None,
        redundancy=None,
        association=(
            [1.0, 0.001, 0.204, 0.176, 0.001, 1.0, 0.002, 0.005]
            + [0.204, 0.002, 1.0, 0.645, 0.176, 0.005, 0.645, 1.0]
        ),
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


def _validate_shap_values_against_predictions(
    shap_values: pd.DataFrame, crossfit: LearnerCrossfit[ClassifierPipelineDF]
):
    # calculate the matching predictions, so we can check if the SHAP values add up
    # correctly
    predicted_probabilities_per_split = [
        model.predict_proba(crossfit.training_sample.features.iloc[test_split, :])
        for model, (_, test_split) in zip(crossfit.models(), crossfit.splits())
    ]
    for split, predicted_probabilities in enumerate(predicted_probabilities_per_split):
        # for each observation, we expect to get the constant "expected probability"
        # value by deducting the SHAP values for all features from the predicted
        # probability
        expected_probability = (
            predicted_probabilities.iloc[:, [0]]
            .join(-shap_values.xs(split, level=0).sum(axis=1).rename("shap"))
            .sum(axis=1)
        )
        min = expected_probability.min()
        max = expected_probability.max()
        assert min == pytest.approx(
            max
        ), "expected probability is the same for all explanations"
        assert 0.4 <= min <= 0.6, "expected class probability is roughly 50%"


def test_model_inspection_classifier_interaction(
    iris_sample_binary: Sample,
    iris_classifier_crossfit: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    model_inspector = ClassifierInspector(n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit
    )
    model_inspector_no_interaction = ClassifierInspector(
        shap_interaction=False, n_jobs=n_jobs
    ).fit(crossfit=iris_classifier_crossfit)

    # calculate shap interaction values
    shap_interaction_values = model_inspector.shap_interaction_values()

    # calculate shap values from interaction values
    shap_values = shap_interaction_values.groupby(by="observation").sum()

    # shap interaction values add up to shap values
    # we have to live with differences of up to 0.006, given the different results
    # returned for SHAP values and SHAP interaction values
    # todo: review accuracy after implementing use of a background dataset
    assert (
        model_inspector_no_interaction.shap_values() - shap_values
    ).abs().max().max() < 0.015

    # the column names of the shap value data frames are the feature names
    feature_columns = iris_sample_binary.feature_columns
    assert shap_values.columns.to_list() == feature_columns
    assert shap_interaction_values.columns.to_list() == feature_columns

    # the length of rows in shap_values should be equal to the number of observations
    assert len(shap_values) == len(iris_sample_binary)

    # the length of rows in shap_interaction_values should be equal to the number of
    # observations, times the number of features
    assert len(shap_interaction_values) == (
        len(iris_sample_binary) * len(feature_columns)
    )

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=model_inspector.shap_interaction_values(consolidate=None)
        .groupby(level=[0, 1])
        .sum(),
        crossfit=iris_classifier_crossfit,
    )

    _check_feature_dependency_matrices(
        model_inspector=model_inspector,
        feature_names=feature_columns,
        synergy=(
            [1.0, 0.018, 0.061, 0.059, 0.018, 1.0, 0.015, 0.015]
            + [0.061, 0.015, 1.0, 0.011, 0.059, 0.015, 0.011, 1.0]
        ),
        redundancy=(
            [1.0, 0.007, 0.221, 0.179, 0.007, 1.0, 0.003, 0.005]
            + [0.221, 0.003, 1.0, 0.651, 0.179, 0.005, 0.651, 1.0]
        ),
        association=(
            [1.0, 0.001, 0.204, 0.176, 0.001, 1.0, 0.002, 0.005]
            + [0.204, 0.002, 1.0, 0.645, 0.176, 0.005, 0.645, 1.0]
        ),
    )

    linkage_tree = model_inspector.feature_redundancy_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature redundancy linkage"
    )


def test_model_inspection_classifier_interaction_dual_target(
    iris_sample_binary_dual_target: Sample,
    iris_classifier_ranker_dual_target: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> None:
    iris_classifier_crossfit_dual_target = (
        iris_classifier_ranker_dual_target.best_model_crossfit
    )

    with pytest.raises(
        ValueError,
        match="only single-output classifiers are supported.*target.*target2",
    ):
        ClassifierInspector(n_jobs=n_jobs).fit(
            crossfit=iris_classifier_crossfit_dual_target
        )


def _check_feature_dependency_matrices(
    model_inspector: ClassifierInspector,
    feature_names: Sequence[str],
    **matrices_expected: Optional[Sequence[float]],
):
    # Shap decomposition matrices (feature dependencies)
    matrix_functions = {
        "synergy": model_inspector.feature_synergy_matrix,
        "redundancy": model_inspector.feature_redundancy_matrix,
        "association": model_inspector.feature_association_matrix,
    }
    for matrix_type, matrix_function in matrix_functions.items():
        matrix_expected = matrices_expected[matrix_type]
        if matrix_expected:
            _check_feature_relationship_matrix(
                matrix=matrix_function(),
                matrix_expected=matrix_expected,
                feature_names=feature_names,
            )
        else:
            with pytest.raises(
                RuntimeError, match="SHAP interaction values have not been calculated"
            ):
                matrix_function()


def _check_feature_relationship_matrix(
    matrix: pd.DataFrame, matrix_expected: Sequence[float], feature_names: Sequence[str]
):
    # check number of rows
    assert len(matrix) == len(feature_names)
    assert len(matrix.columns) == len(feature_names)

    # check association values
    assert 0.0 <= matrix.min().min() <= matrix.max().max() <= 1.0

    assert matrix.values.ravel() == pytest.approx(matrix_expected, abs=1e-2)
