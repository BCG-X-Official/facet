"""
Model inspector tests.
"""
import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.model_selection import BaseCrossValidator, KFold

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import (
    ClassifierInspector,
    KernelExplainerFactory,
    RegressorInspector,
)
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
    cv_kfold: KFold,
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs: int,
    fast_execution: bool,
) -> None:

    # define checksums for this test
    if fast_execution:
        expected_scores = [0.418, 0.4, 0.386, 0.385, 0.122] + [
            0.122,
            -0.074,
            -0.074,
            -0.074,
            -0.074,
        ]
    else:
        expected_scores = [0.202, 0.160, 0.111, 0.056, 0.031] + [
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
    shap_values_mean = regressor_inspector.shap_values(consolidate="mean")
    shap_values_std = regressor_inspector.shap_values(consolidate="std")

    # method shap_values without parameter is equal to "mean" consolidation
    assert_frame_equal(shap_values_mean, regressor_inspector.shap_values())

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values_mean) == len(sample)

    # index names
    assert shap_values_mean.index.names == [Sample.COL_OBSERVATION]
    assert shap_values_mean.columns.names == [Sample.COL_FEATURE]
    assert shap_values_std.index.names == [Sample.COL_OBSERVATION]
    assert shap_values_std.columns.names == [Sample.COL_FEATURE]
    assert shap_values_raw.index.names == (
        [RegressorInspector.COL_SPLIT, Sample.COL_OBSERVATION]
    )
    assert shap_values_raw.columns.names == [Sample.COL_FEATURE]

    # column index
    assert set(shap_values_mean.columns) == feature_names

    # check that the SHAP values add up to the predictions
    shap_totals_raw = shap_values_raw.sum(axis=1)

    for split_id, model in enumerate(best_lgbm_crossfit.models()):
        # for each model in the crossfit, calculate the difference between total
        # SHAP values and prediction for every observation. This is always the same
        # constant value, so `mad` (mean absolute deviation) is zero

        shap_minus_pred = shap_totals_raw.xs(key=split_id) - model.predict(
            X=sample.features
        )
        assert (
            round(shap_minus_pred.mad(), 12) == 0.0
        ), f"predictions matching total SHAP for split {split_id}"

    #  test the ModelInspector with a KernelExplainer:

    inspector_2 = RegressorInspector(
        explainer_factory=KernelExplainerFactory(link="identity", data_size_limit=20),
        n_jobs=n_jobs,
    ).fit(crossfit=best_lgbm_crossfit)
    inspector_2.shap_values()

    linkage_tree = inspector_2.feature_association_linkage()

    print()
    DendrogramDrawer(style="text").draw(data=linkage_tree, title="Test")


def test_binary_classifier_ranking(iris_classifier_ranker_binary) -> None:

    expected_learner_scores = [0.858, 0.82, 0.546, 0.287]

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


# noinspection DuplicatedCode
def test_model_inspection_classifier_binary(
    iris_sample_binary: Sample, iris_classifier_crossfit_binary, n_jobs: int
) -> None:

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

    assert model_inspector.feature_association_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.005, 0.197, 0.168],
                [0.005, 1.0, 0.002, 0.004],
                [0.197, 0.002, 1.0, 0.644],
                [0.168, 0.004, 0.644, 1.0],
            ]
        ),
        abs=0.02,
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


# noinspection DuplicatedCode
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

    assert model_inspector.feature_synergy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.039, 0.042, 0.03, 1.0, 0.057]
                + [0.125, 0.113, 1.0, 0.027, 0.093, 0.088],
                [0.039, 1.0, 0.013, 0.015, 0.057, 1.0]
                + [0.018, 0.027, 0.027, 1.0, 0.014, 0.018],
                [0.042, 0.013, 1.0, 0.055, 0.125, 0.018]
                + [1.0, 0.244, 0.093, 0.014, 1.0, 0.166],
                [0.03, 0.015, 0.055, 1.0, 0.113, 0.027]
                + [0.244, 1.0, 0.088, 0.018, 0.166, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert model_inspector.feature_redundancy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.072, 0.478, 0.45, 1.0, 0.111]
                + [0.162, 0.157, 1.0, 0.006, 0.358, 0.339],
                [0.072, 1.0, 0.049, 0.047, 0.111, 1.0]
                + [0.036, 0.041, 0.006, 1.0, 0.002, 0.001],
                [0.478, 0.049, 1.0, 0.901, 0.162, 0.036]
                + [1.0, 0.504, 0.358, 0.002, 1.0, 0.693],
                [0.45, 0.047, 0.901, 1.0, 0.157, 0.041]
                + [0.504, 1.0, 0.339, 0.001, 0.693, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert model_inspector.feature_association_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.062, 0.471, 0.447, 1.0, 0.102]
                + [0.238, 0.236, 1.0, 0.006, 0.344, 0.33],
                [0.062, 1.0, 0.05, 0.049, 0.102, 1.0]
                + [0.041, 0.046, 0.006, 1.0, 0.002, 0.003],
                [0.471, 0.05, 1.0, 0.923, 0.238, 0.041]
                + [1.0, 0.709, 0.344, 0.002, 1.0, 0.705],
                [0.447, 0.049, 0.923, 1.0, 0.236, 0.046]
                + [0.709, 1.0, 0.33, 0.003, 0.705, 1.0],
            ]
        ),
        abs=0.02,
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
            assert expected_proba_range * 0.6 <= min <= expected_proba_range / 0.6, (
                "expected class probability is roughly in the range of "
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

    assert model_inspector.feature_synergy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.013, 0.061, 0.038],
                [0.013, 1.0, 0.014, 0.017],
                [0.061, 0.014, 1.0, 0.013],
                [0.038, 0.017, 0.013, 1.0],
            ]
        ),
        abs=0.02,
    )

    assert model_inspector.feature_redundancy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.006, 0.207, 0.177],
                [0.006, 1.0, 0.002, 0.002],
                [0.207, 0.002, 1.0, 0.651],
                [0.177, 0.002, 0.651, 1.0],
            ]
        ),
        abs=0.02,
    )

    assert model_inspector.feature_association_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.005, 0.197, 0.168],
                [0.005, 1.0, 0.002, 0.004],
                [0.197, 0.002, 1.0, 0.644],
                [0.168, 0.004, 0.644, 1.0],
            ]
        ),
        abs=0.02,
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
