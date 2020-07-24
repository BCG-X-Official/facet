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

    # define checksums for this test
    checksum_shap = 4149000024927268877
    checksum_matrices = {
        "synergy": None,
        "redundancy": None,
        "association": 9024604747537243391,
    }
    model_inspector = ClassifierInspector(shap_interaction=False, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit
    )
    # make and check shap value matrix
    shap_matrix = model_inspector.shap_values()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values)
        == checksum_shap
    )

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix) == len(iris_sample_binary)

    # Shap decomposition matrices (feature dependencies)
    feature_associations: pd.DataFrame = model_inspector.feature_association_matrix()

    _check_feature_dependency_matrices(
        model_inspector=model_inspector,
        feature_names=iris_sample_binary.feature_columns,
        **checksum_matrices,
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


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

    # define checksums for this test
    checksum_shap = 11537888084826694975
    checksum_matrices = {
        "synergy": 18268622212148004967,
        "redundancy": 2146036076598277859,
        "association": 10618889851519629191,
    }
    model_inspector = ClassifierInspector(n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit
    )
    model_inspector_no_interaction = ClassifierInspector(
        shap_interaction=False, n_jobs=n_jobs
    ).fit(crossfit=iris_classifier_crossfit)
    # calculate shap interaction values
    shap_interaction_matrix = model_inspector.shap_interaction_values()

    # shap interaction values add up to shap values
    # we have to live with differences of up to 0.006, given the different results
    # returned for SHAP values and SHAP interaction values
    # todo: review accuracy after implementing use of a background dataset
    assert (
        model_inspector_no_interaction.shap_values()
        - shap_interaction_matrix.groupby(by="observation").sum()
    ).abs().max().max() < 0.015

    # the length of rows in shap_values should be equal to the number of observations,
    # times the number of features
    feature_names = iris_sample_binary.feature_columns
    assert len(shap_interaction_matrix) == (
        len(iris_sample_binary) * len(feature_names)
    )

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_interaction_matrix.round(decimals=4)).values)
        == checksum_shap
    )

    _check_feature_dependency_matrices(
        model_inspector=model_inspector,
        feature_names=feature_names,
        **checksum_matrices,
    )

    linkage_tree = model_inspector.feature_redundancy_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature redundancy linkage"
    )


def _check_feature_dependency_matrices(
    model_inspector: ClassifierInspector,
    feature_names: Sequence[str],
    **checksums_expected: Optional[int],
):
    # Shap decomposition matrices (feature dependencies)
    matrix_functions = {
        "synergy": model_inspector.feature_synergy_matrix,
        "redundancy": model_inspector.feature_redundancy_matrix,
        "association": model_inspector.feature_association_matrix,
    }
    for matrix_type, matrix_function in matrix_functions.items():
        checksum_expected = checksums_expected[matrix_type]
        if checksum_expected:
            _check_feature_relationship_matrix(
                matrix=matrix_function(),
                checksum_expected=checksum_expected,
                feature_names=feature_names,
            )
        else:
            with pytest.raises(
                RuntimeError, match="SHAP interaction values have not been calculated"
            ):
                matrix_function()


def _check_feature_relationship_matrix(
    matrix: pd.DataFrame, checksum_expected: int, feature_names: Sequence[str]
):
    # check number of rows
    assert len(matrix) == len(feature_names)
    assert len(matrix.columns) == len(feature_names)

    # check association values
    for c in matrix.columns:
        fa = matrix.loc[:, c]
        assert 0.0 <= fa.min() <= fa.max() <= 1.0
    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(matrix.round(decimals=4)).values) == checksum_expected
    )
