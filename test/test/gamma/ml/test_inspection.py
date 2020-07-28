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
from sklearn.model_selection import BaseCrossValidator

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import ClassifierInspector, RegressorInspector
from gamma.ml.selection import LearnerRanker, ParameterGrid
from gamma.ml.validation import BootstrapCV
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from test.gamma.ml import check_ranking, disable_warnings

# noinspection PyMissingOrEmptyDocstring

log = logging.getLogger(__name__)
disable_warnings()


@pytest.fixture
def iris_classifier_ranker_binary(
    iris_sample_binary: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample_binary, cv=cv_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_multi_class(
    iris_sample: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(sample=iris_sample, cv=cv_bootstrap, n_jobs=n_jobs)


@pytest.fixture
def iris_classifier_ranker_dual_target(
    iris_sample_binary_dual_target: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample_binary_dual_target, cv=cv_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_crossfit_binary(
    iris_classifier_ranker_binary: LearnerRanker[ClassifierPipelineDF]
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_binary.best_model_crossfit


@pytest.fixture
def iris_classifier_crossfit_multi_class(
    iris_classifier_ranker_multi_class: LearnerRanker[ClassifierPipelineDF]
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_multi_class.best_model_crossfit


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


def test_binary_classifier_ranking(iris_classifier_ranker_binary) -> None:

    expected_learner_scores = [0.858, 0.835, 0.784, 0.689]

    log.debug(f"\n{iris_classifier_ranker_binary.summary_report(max_learners=10)}")
    check_ranking(
        ranking=iris_classifier_ranker_binary.ranking(),
        expected_scores=expected_learner_scores,
        expected_learners=[RandomForestClassifierDF] * 4,
        expected_parameters={
            2: dict(classifier__min_samples_leaf=32, classifier__n_estimators=50),
            3: dict(classifier__min_samples_leaf=32, classifier__n_estimators=10),
        },
    )


def test_model_inspection_classifier_binary(
    iris_sample_binary: Sample, iris_classifier_crossfit_binary, n_jobs: int
) -> None:
    disable_warnings()

    model_inspector = ClassifierInspector(shap_interaction=False, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_binary
    )

    # calculate the shap value matrix, without any consolidation
    shap_values = model_inspector.shap_values(consolidate=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit_binary
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
        n_classes=1,
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


def test_model_inspection_classifier_multi_class(
    iris_sample: Sample,
    iris_classifier_crossfit_multi_class: LearnerCrossfit[ClassifierPipelineDF],
    n_jobs: int,
) -> None:

    model_inspector = ClassifierInspector(shap_interaction=True, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_multi_class
    )

    # calculate the shap value matrix, without any consolidation
    shap_values = model_inspector.shap_values(consolidate=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit_multi_class
    )

    shap_matrix_mean = model_inspector.shap_values()

    # is the consolidation correct?
    assert_frame_equal(shap_matrix_mean, shap_values.mean(level=1))

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix_mean) == len(iris_sample)

    # Shap decomposition matrices (feature dependencies)
    _check_feature_dependency_matrices(
        model_inspector=model_inspector,
        feature_names=iris_sample.feature_columns,
        n_classes=3,
        synergy=(
            [1.0, 0.06, 0.065, 0.042, 1.0, 0.079, 0.113, 0.125]
            + [1.0, 0.01, 0.062, 0.095, 0.06, 1.0, 0.019, 0.017]
            + [0.079, 1.0, 0.025, 0.019, 0.01, 1.0, 0.017, 0.003]
            + [0.065, 0.019, 1.0, 0.102, 0.113, 0.025, 1.0, 0.252]
            + [0.062, 0.017, 1.0, 0.153, 0.042, 0.017, 0.102, 1.0]
            + [0.125, 0.019, 0.252, 1.0, 0.095, 0.003, 0.153, 1.0]
        ),
        redundancy=(
            [1.0, 0.133, 0.443, 0.416, 1.0, 0.13, 0.187, 0.16]
            + [1.0, 0.01, 0.375, 0.422, 0.133, 1.0, 0.072, 0.067]
            + [0.13, 1.0, 0.045, 0.064, 0.01, 1.0, 0.004, 0.004]
            + [0.443, 0.072, 1.0, 0.861, 0.187, 0.045, 1.0, 0.497]
            + [0.375, 0.004, 1.0, 0.685, 0.416, 0.067, 0.861, 1.0]
            + [0.16, 0.064, 0.497, 1.0, 0.422, 0.004, 0.685, 1.0]
        ),
        association=(
            [1.0, 0.122, 0.444, 0.418, 1.0, 0.13, 0.256, 0.249]
            + [1.0, 0.009, 0.366, 0.412, 0.122, 1.0, 0.075, 0.072]
            + [0.13, 1.0, 0.054, 0.069, 0.009, 1.0, 0.005, 0.004]
            + [0.444, 0.075, 1.0, 0.924, 0.256, 0.054, 1.0, 0.712]
            + [0.366, 0.005, 1.0, 0.695, 0.418, 0.072, 0.924, 1.0]
            + [0.249, 0.069, 0.712, 1.0, 0.412, 0.004, 0.695, 1.0]
        ),
    )

    linkage_trees = model_inspector.feature_association_linkage()

    for output, linkage_tree in zip(model_inspector.outputs, linkage_trees):
        print()
        DendrogramDrawer(style=DendrogramReportStyle()).draw(
            data=linkage_tree, title=f"Iris feature association linkage: {output}"
        )


def _validate_shap_values_against_predictions(
    shap_values: pd.DataFrame, crossfit: LearnerCrossfit[ClassifierPipelineDF]
):

    # calculate the matching predictions, so we can check if the SHAP values add up
    # correctly
    predicted_probabilities_per_split: List[pd.DataFrame] = [
        model.predict_proba(crossfit.training_sample.features.iloc[test_split, :])
        for model, (_, test_split) in zip(crossfit.models(), crossfit.splits())
    ]

    for split, predicted_probabilities in enumerate(predicted_probabilities_per_split):

        assert isinstance(
            predicted_probabilities, pd.DataFrame
        ), "predicted probabilities are single-output"

        expected_proba_range = 1 / len(predicted_probabilities.columns)

        def _check_probabilities(
            _class_probabilities: pd.DataFrame, _shap_for_split_and_class: pd.Series
        ) -> None:
            expected_probability = _class_probabilities.join(
                _shap_for_split_and_class
            ).sum(axis=1)

            min = expected_probability.min()
            max = expected_probability.max()
            assert min == pytest.approx(
                max
            ), "expected probability is the same for all explanations"
            assert expected_proba_range * 0.75 <= min <= expected_proba_range / 0.75, (
                "expected class probability is roughly "
                f"{expected_proba_range * 100:.0f}%"
            )

        if predicted_probabilities.shape[1] == 2:
            # for binary classification we have SHAP values only for the first class
            _check_probabilities(
                predicted_probabilities.iloc[:, [0]],
                -shap_values.xs(split).sum(axis=1).rename("shap"),
            )

        else:
            # multi-class classification has outputs for each class

            for class_name in predicted_probabilities.columns:
                # for each observation and class, we expect to get the constant
                # "expected probability" value by deducting the SHAP values for all features
                # from the predicted probability

                class_probabilities = predicted_probabilities.loc[:, [class_name]]

                shap_for_split_and_class = (
                    -shap_values.xs(split)
                    .xs(class_name, axis=1)
                    .sum(axis=1)
                    .rename("shap")
                )

                _check_probabilities(class_probabilities, shap_for_split_and_class)


def test_model_inspection_classifier_interaction(
    iris_sample_binary: Sample, iris_classifier_crossfit_binary, n_jobs: int
) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    model_inspector = ClassifierInspector(n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_binary
    )
    model_inspector_no_interaction = ClassifierInspector(
        shap_interaction=False, n_jobs=n_jobs
    ).fit(crossfit=iris_classifier_crossfit_binary)

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
        crossfit=iris_classifier_crossfit_binary,
    )

    _check_feature_dependency_matrices(
        model_inspector=model_inspector,
        feature_names=feature_columns,
        n_classes=1,
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


def _fit_learner_ranker(
    sample: Sample, cv: BaseCrossValidator, n_jobs: int
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
        cv=cv,
        scoring="f1_macro",
        # shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=sample)


def _check_feature_dependency_matrices(
    model_inspector: ClassifierInspector,
    feature_names: Sequence[str],
    n_classes: int,
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
                n_classes=n_classes,
            )
        else:
            with pytest.raises(
                RuntimeError, match="SHAP interaction values have not been calculated"
            ):
                matrix_function()


def _check_feature_relationship_matrix(
    matrix: pd.DataFrame,
    matrix_expected: Sequence[float],
    feature_names: Sequence[str],
    n_classes: int,
):
    # check number of rows
    assert matrix.shape == (len(feature_names), len(feature_names) * n_classes)

    # check association values
    assert 0.0 <= matrix.min().min() <= matrix.max().max() <= 1.0

    assert matrix.values.ravel() == pytest.approx(matrix_expected, abs=1e-2)
