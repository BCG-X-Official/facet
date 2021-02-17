"""
Model inspector tests.
"""
import logging
import warnings
from typing import Any, Callable, List, Sequence, Set, Tuple, TypeVar

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.datasets import make_classification
from sklearn.model_selection import BaseCrossValidator, KFold

from pytools.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from sklearndf import TransformerDF
from sklearndf.classification import (
    GradientBoostingClassifierDF,
    RandomForestClassifierDF,
)
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF

from ..conftest import check_ranking
from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.inspection import (
    KernelExplainerFactory,
    LearnerInspector,
    TreeExplainerFactory,
)
from facet.selection import LearnerGrid, LearnerRanker
from facet.validation import BootstrapCV, StratifiedBootstrapCV

# noinspection PyMissingOrEmptyDocstring

log = logging.getLogger(__name__)

T = TypeVar("T")


@pytest.fixture
def iris_classifier_ranker_binary(
    iris_sample_binary: Sample,
    cv_stratified_bootstrap: StratifiedBootstrapCV,
    n_jobs: int,
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample_binary, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_multi_class(
    iris_sample: Sample, cv_stratified_bootstrap: StratifiedBootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_dual_target(
    iris_sample_binary_dual_target: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample_binary_dual_target, cv=cv_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_crossfit_binary(
    iris_classifier_ranker_binary: LearnerRanker[ClassifierPipelineDF],
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_binary.best_model_crossfit_


@pytest.fixture
def iris_classifier_crossfit_multi_class(
    iris_classifier_ranker_multi_class: LearnerRanker[ClassifierPipelineDF],
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_multi_class.best_model_crossfit_


@pytest.fixture
def iris_inspector_multi_class(
    iris_classifier_crossfit_multi_class: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> LearnerInspector[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return LearnerInspector(shap_interaction=True, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_multi_class
    )


def test_model_inspection(
    regressor_grids: Sequence[LearnerGrid[RegressorPipelineDF]],
    regressor_ranker: LearnerRanker[RegressorPipelineDF],
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF],
    feature_names: Set[str],
    regressor_inspector: LearnerInspector,
    cv_kfold: KFold,
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs: int,
) -> None:

    # define checksums for this test
    expected_scores = [0.418, 0.4, 0.386, 0.385, 0.122] + [
        0.122,
        -0.074,
        -0.074,
        -0.074,
        -0.074,
    ]

    log.debug(f"\n{regressor_ranker.summary_report()}")

    check_ranking(
        ranking=regressor_ranker.ranking_,
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
    assert shap_values_mean.index.names == [Sample.IDX_OBSERVATION]
    assert shap_values_mean.columns.names == [Sample.IDX_FEATURE]
    assert shap_values_std.index.names == [Sample.IDX_OBSERVATION]
    assert shap_values_std.columns.names == [Sample.IDX_FEATURE]
    assert shap_values_raw.index.names == ["split", "observation"]
    assert shap_values_raw.columns.names == [Sample.IDX_FEATURE]

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

    inspector_2 = LearnerInspector(
        explainer_factory=KernelExplainerFactory(link="identity", data_size_limit=20),
        n_jobs=n_jobs,
    ).fit(crossfit=best_lgbm_crossfit)
    inspector_2.shap_values()

    linkage_tree = inspector_2.feature_association_linkage()

    print()
    DendrogramDrawer(style="text").draw(data=linkage_tree, title="Test")


def test_binary_classifier_ranking(iris_classifier_ranker_binary) -> None:

    expected_learner_scores = [0.872, 0.868, 0.866, 0.859]

    log.debug(f"\n{iris_classifier_ranker_binary.summary_report()}")
    check_ranking(
        ranking=iris_classifier_ranker_binary.ranking_,
        expected_scores=expected_learner_scores,
        expected_learners=[RandomForestClassifierDF] * 4,
        expected_parameters={
            2: dict(classifier__min_samples_leaf=4, classifier__n_estimators=10),
            3: dict(classifier__min_samples_leaf=8, classifier__n_estimators=10),
        },
    )


# noinspection DuplicatedCode
def test_model_inspection_classifier_binary(
    iris_sample_binary: Sample, iris_classifier_crossfit_binary, n_jobs: int
) -> None:

    model_inspector = LearnerInspector(shap_interaction=False, n_jobs=n_jobs).fit(
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

    (
        association_matrix,
        association_matrix_legacy,
    ) = call_inspector_method_both_algorithms(
        model_inspector.feature_association_matrix,
        clustered=True,
        symmetrical=True,
    )
    assert association_matrix.values == pytest.approx(
        np.array(
            [
                [1.000, 0.671, 0.212, 0.005],
                [0.671, 1.000, 0.331, 0.014],
                [0.212, 0.331, 1.000, 0.006],
                [0.006, 0.014, 0.006, 1.000],
            ]
        ),
        abs=0.02,
    )
    assert association_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.000, 0.678, 0.133, 0.005],
                [0.678, 1.000, 0.145, 0.007],
                [0.133, 0.145, 1.000, 0.029],
                [0.005, 0.007, 0.029, 1.000],
            ]
        ),
        abs=0.02,
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


def test_model_inspection_classifier_binary_single_shap_output() -> None:
    # simulate some data
    x, y = make_classification(
        n_samples=200, n_features=5, n_informative=5, n_redundant=0, random_state=42
    )
    sim_df = pd.DataFrame(
        np.hstack([x, y[:, np.newaxis]]),
        columns=[*(f"f{i}" for i in range(5)), "target"],
    )

    # create sample object
    sample_df = Sample(observations=sim_df, target_name="target")

    # fit the crossfit
    crossfit = LearnerCrossfit(
        pipeline=ClassifierPipelineDF(
            classifier=GradientBoostingClassifierDF(random_state=42)
        ),
        cv=BootstrapCV(n_splits=5, random_state=42),
        random_state=42,
        n_jobs=-3,
    ).fit(sample_df)

    # fit the inspector
    LearnerInspector(n_jobs=-3).fit(crossfit=crossfit)


# noinspection DuplicatedCode
def test_model_inspection_classifier_multi_class(
    iris_sample: Sample,
    iris_classifier_crossfit_multi_class: LearnerCrossfit[ClassifierPipelineDF],
    iris_inspector_multi_class: LearnerInspector[ClassifierPipelineDF],
    n_jobs: int,
) -> None:

    # calculate the shap value matrix, without any consolidation
    shap_values = iris_inspector_multi_class.shap_values(consolidate=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit_multi_class
    )

    shap_matrix_mean: List[pd.DataFrame] = iris_inspector_multi_class.shap_values()

    for _mean, _raw in zip(shap_matrix_mean, shap_values):
        # is the consolidation correct?
        assert_frame_equal(_mean, _raw.mean(level=1))

        # the length of rows in shap_values should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(_mean) == len(iris_sample)

    # Feature importance

    feature_importance: pd.DataFrame = iris_inspector_multi_class.feature_importance()
    assert feature_importance.index.equals(
        pd.Index(iris_sample.feature_names, name="feature")
    )
    assert feature_importance.columns.equals(
        pd.Index(iris_inspector_multi_class.output_names_, name="class")
    )
    assert feature_importance.values == pytest.approx(
        np.array(
            [
                [0.125, 0.085, 0.104],
                [0.020, 0.019, 0.010],
                [0.424, 0.456, 0.461],
                [0.432, 0.441, 0.425],
            ]
        ),
        abs=0.02,
    )

    # Shap decomposition matrices (feature dependencies)

    synergy_matrix, synergy_matrix_legacy = call_inspector_method_both_algorithms(
        iris_inspector_multi_class.feature_synergy_matrix,
        clustered=False,
        symmetrical=False,
    )
    assert np.hstack([m.values for m in synergy_matrix_legacy]) == pytest.approx(
        np.array(
            [
                [1.0, 0.04, 0.149, 0.124, 1.0, 0.059]
                + [0.345, 0.323, 1.0, 0.011, 0.213, 0.204],
                [0.196, 1.0, 0.116, 0.119, 0.209, 1.0]
                + [0.204, 0.225, 0.104, 1.0, 0.297, 0.306],
                [0.053, 0.006, 1.0, 0.022, 0.077, 0.01]
                + [1.0, 0.196, 0.066, 0.008, 1.0, 0.157],
                [0.042, 0.006, 0.021, 1.0, 0.069, 0.011]
                + [0.203, 1.0, 0.067, 0.009, 0.167, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert np.hstack([m.values for m in synergy_matrix]) == pytest.approx(
        np.array(
            [
                [1.0, 0.012, 0.157, 0.155, 1.0, 0.148]
                + [0.868, 0.869, 1.0, 0.0, 0.828, 0.849],
                [0.082, 1.0, 0.058, 0.09, 0.077, 1.0]
                + [0.204, 0.247, 0.04, 1.0, 0.358, 0.384],
                [0.004, 0.0, 1.0, 0.002, 0.029, 0.004]
                + [1.0, 0.04, 0.006, 0.001, 1.0, 0.014],
                [0.001, 0.0, 0.001, 1.0, 0.027, 0.001]
                + [0.037, 1.0, 0.014, 0.004, 0.026, 1.0],
            ]
        ),
        abs=0.02,
    )

    redundancy_matrix, redundancy_matrix_legacy = call_inspector_method_both_algorithms(
        iris_inspector_multi_class.feature_redundancy_matrix,
        clustered=False,
        symmetrical=False,
    )
    assert np.hstack([m.values for m in redundancy_matrix_legacy]) == (
        pytest.approx(
            np.array(
                [
                    [1.0, 0.077, 0.67, 0.667, 1.0, 0.084]
                    + [0.37, 0.352, 1.0, 0.006, 0.671, 0.624],
                    [0.356, 1.0, 0.45, 0.447, 0.297, 1.0]
                    + [0.306, 0.304, 0.054, 1.0, 0.026, 0.086],
                    [0.261, 0.028, 1.0, 0.97, 0.084, 0.016]
                    + [1.0, 0.583, 0.197, 0.001, 1.0, 0.706],
                    [0.254, 0.028, 0.96, 1.0, 0.082, 0.016]
                    + [0.591, 1.0, 0.202, 0.002, 0.741, 1.0],
                ]
            ),
            abs=0.02,
        )
    )
    assert np.hstack([m.values for m in redundancy_matrix]) == (
        pytest.approx(
            np.array(
                [
                    [1.0, 0.099, 0.551, 0.561, 1.0, 0.034]
                    + [0.024, 0.016, 1.0, 0.031, 0.016, 0.037],
                    [0.092, 1.0, 0.371, 0.37, 0.037, 1.0]
                    + [0.079, 0.115, 0.03, 1.0, 0.002, 0.005],
                    [0.652, 0.393, 1.0, 0.997, 0.179, 0.099]
                    + [1.0, 0.734, 0.093, 0.003, 1.0, 0.819],
                    [0.663, 0.406, 0.999, 1.0, 0.121, 0.152]
                    + [0.736, 1.0, 0.242, 0.007, 0.81, 1.0],
                ]
            ),
            abs=0.02,
        )
    )

    (
        association_matrix,
        association_matrix_legacy,
    ) = call_inspector_method_both_algorithms(
        iris_inspector_multi_class.feature_association_matrix,
        clustered=False,
        symmetrical=False,
    )
    assert np.hstack([m.values for m in association_matrix_legacy]) == (
        pytest.approx(
            np.array(
                [
                    [1.0, 0.049, 0.632, 0.635, 1.0, 0.053]
                    + [0.474, 0.458, 1.0, -0.008, 0.626, 0.578],
                    [0.258, 1.0, 0.455, 0.45, 0.206, 1.0]
                    + [0.332, 0.371, -0.082, 1.0, -0.027, -0.02],
                    [0.235, 0.029, 1.0, 0.983, 0.116, 0.018]
                    + [1.0, 0.684, 0.183, -0.001, 1.0, 0.665],
                    [0.233, 0.028, 0.972, 1.0, 0.116, 0.021]
                    + [0.7, 1.0, 0.184, -0.0, 0.702, 1.0],
                ]
            ),
            abs=0.02,
        )
    )
    assert np.hstack([m.values for m in association_matrix]) == (
        pytest.approx(
            np.array(
                [
                    [1.0, 0.101, 0.666, 0.674, 1.0, 0.06]
                    + [0.4, 0.371, 1.0, 0.006, 0.67, 0.579],
                    [0.101, 1.0, 0.397, 0.388, 0.06, 1.0]
                    + [0.192, 0.25, 0.006, 1.0, 0.001, 0.0],
                    [0.666, 0.397, 1.0, 0.999, 0.4, 0.192]
                    + [1.0, 0.796, 0.67, 0.001, 1.0, 0.787],
                    [0.674, 0.388, 0.999, 1.0, 0.371, 0.25]
                    + [0.796, 1.0, 0.579, 0.0, 0.787, 1.0],
                ]
            ),
            abs=0.02,
        )
    )

    linkage_trees = iris_inspector_multi_class.feature_association_linkage()

    for output, linkage_tree in zip(
        iris_inspector_multi_class.output_names_, linkage_trees
    ):
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
        model.predict_proba(crossfit.sample_.features.iloc[test_split, :])
        for model, (_, test_split) in zip(crossfit.models(), crossfit.splits())
    ]

    for split, predicted_probabilities in enumerate(predicted_probabilities_per_split):

        assert isinstance(
            predicted_probabilities, pd.DataFrame
        ), "predicted probabilities are single-output"

        expected_probability_range = 1 / len(predicted_probabilities.columns)

        def _check_probabilities(
            _class_probabilities: pd.DataFrame, _shap_for_split_and_class: pd.Series
        ) -> None:
            expected_probability = _class_probabilities.join(
                _shap_for_split_and_class
            ).sum(axis=1)

            expected_probability_min = expected_probability.min()
            expected_probability_max = expected_probability.max()
            assert expected_probability_min == pytest.approx(
                expected_probability_max
            ), "expected probability is the same for all explanations"
            assert (
                expected_probability_range * 0.6
                <= expected_probability_min
                <= expected_probability_range / 0.6
            ), (
                "expected class probability is roughly in the range of "
                f"{expected_probability_range * 100:.0f}%"
            )

        if predicted_probabilities.shape[1] == 2:
            # for binary classification we have SHAP values only for the first class
            _check_probabilities(
                predicted_probabilities.iloc[:, [0]],
                -shap_values.xs(split).sum(axis=1).rename("shap"),
            )

        else:
            # multi-class classification has outputs for each class

            for class_idx, class_name in enumerate(predicted_probabilities.columns):
                # for each observation and class, we expect to get the constant
                # expected probability value by deducting the SHAP values for all
                # features from the predicted probability

                class_probabilities = predicted_probabilities.loc[:, [class_name]]

                shap_for_split_and_class = (
                    -shap_values[class_idx].xs(split).sum(axis=1).rename("shap")
                )

                _check_probabilities(class_probabilities, shap_for_split_and_class)


# noinspection DuplicatedCode
def test_model_inspection_classifier_interaction(
    iris_sample_binary: Sample,
    iris_classifier_crossfit_binary: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> None:
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    model_inspector = LearnerInspector(
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", use_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(crossfit=iris_classifier_crossfit_binary)

    model_inspector_no_interaction = LearnerInspector(
        shap_interaction=False,
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", use_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(crossfit=iris_classifier_crossfit_binary)

    # calculate shap interaction values
    shap_interaction_values = model_inspector.shap_interaction_values()

    # calculate shap values from interaction values
    shap_values = shap_interaction_values.groupby(by="observation").sum()

    # shap interaction values add up to shap values
    # we have to live with differences of up to 0.02, given the different results
    # returned for SHAP values and SHAP interaction values
    # todo: review accuracy after implementing use of a background dataset
    assert (
        model_inspector_no_interaction.shap_values() - shap_values
    ).abs().max().max() < 0.015

    # the column names of the shap value data frames are the feature names
    feature_columns = iris_sample_binary.feature_names
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

    synergy_matrix, synergy_matrix_legacy = call_inspector_method_both_algorithms(
        model_inspector.feature_synergy_matrix, clustered=False, symmetrical=True
    )
    assert synergy_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.000, 0.047, 0.101, 0.120],
                [0.047, 1.000, 0.017, 0.021],
                [0.101, 0.017, 1.000, 0.100],
                [0.120, 0.021, 0.100, 1.000],
            ]
        ),
        abs=0.02,
    )
    assert synergy_matrix.values == pytest.approx(
        np.array(
            [
                [1.000, 0.000, 0.005, 0.007],
                [0.000, 1.000, 0.004, 0.006],
                [0.005, 0.004, 1.000, 0.001],
                [0.007, 0.006, 0.001, 1.000],
            ]
        ),
        abs=0.02,
    )

    synergy_matrix, synergy_matrix_legacy = call_inspector_method_both_algorithms(
        model_inspector.feature_synergy_matrix, clustered=True, symmetrical=False
    )
    assert synergy_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.0, 0.058, 0.091, 0.008],
                [0.26, 1.0, 0.257, 0.024],
                [0.103, 0.078, 1.0, 0.011],
                [0.28, 0.104, 0.297, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert synergy_matrix.values == pytest.approx(
        np.array(
            [
                [1.0, 0.157, 0.001, 0.135],
                [0.0, 1.0, 0.0, 0.001],
                [0.0, 0.108, 1.0, 0.101],
                [0.0, 0.001, 0.0, 1.0],
            ]
        ),
        abs=0.02,
    )

    redundancy_matrix, redundancy_matrix_legacy = call_inspector_method_both_algorithms(
        model_inspector.feature_redundancy_matrix, clustered=False, symmetrical=True
    )
    assert redundancy_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.0, 0.039, 0.181, 0.206],
                [0.039, 1.0, 0.005, 0.011],
                [0.181, 0.005, 1.0, 0.792],
                [0.206, 0.011, 0.792, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert redundancy_matrix.values == pytest.approx(
        np.array(
            [
                [1.0, 0.034, 0.304, 0.217],
                [0.034, 1.0, 0.111, 0.084],
                [0.304, 0.111, 1.0, 0.733],
                [0.217, 0.084, 0.733, 1.0],
            ]
        ),
        abs=0.02,
    )

    redundancy_matrix, redundancy_matrix_legacy = call_inspector_method_both_algorithms(
        model_inspector.feature_redundancy_matrix, clustered=True, symmetrical=False
    )
    assert redundancy_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.0, 0.655, 0.098, 0.002],
                [0.7, 1.0, 0.111, 0.006],
                [0.526, 0.494, 1.0, 0.021],
                [0.081, 0.152, 0.092, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert redundancy_matrix.values == pytest.approx(
        np.array(
            [
                [1.0, 0.733, 0.218, 0.085],
                [0.733, 1.0, 0.305, 0.111],
                [0.194, 0.274, 1.0, 0.034],
                [0.071, 0.096, 0.034, 1.0],
            ]
        ),
        abs=0.02,
    )

    (
        association_matrix,
        association_matrix_legacy,
    ) = call_inspector_method_both_algorithms(
        model_inspector.feature_association_matrix, clustered=False, symmetrical=True
    )
    assert association_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.0, 0.028, 0.14, 0.128],
                [0.028, 1.0, 0.005, 0.002],
                [0.14, 0.005, 1.0, 0.681],
                [0.128, 0.002, 0.681, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert association_matrix.values == pytest.approx(
        np.array(
            [
                [1.0, 0.005, 0.354, 0.227],
                [0.005, 1.0, 0.008, 0.001],
                [0.354, 0.008, 1.0, 0.676],
                [0.227, 0.001, 0.676, 1.0],
            ]
        ),
        abs=0.02,
    )

    (
        association_matrix,
        association_matrix_legacy,
    ) = call_inspector_method_both_algorithms(
        model_inspector.feature_association_matrix, clustered=True, symmetrical=False
    )
    assert association_matrix_legacy.values == pytest.approx(
        np.array(
            [
                [1.0, 0.631, 0.069, -0.001],
                [0.576, 1.0, 0.076, -0.002],
                [0.365, 0.442, 1.0, -0.014],
                [-0.029, -0.096, -0.07, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert association_matrix.values == pytest.approx(
        np.array(
            [
                [1.0, 0.676, 0.227, 0.001],
                [0.676, 1.0, 0.354, 0.008],
                [0.227, 0.354, 1.0, 0.005],
                [0.001, 0.008, 0.005, 1.0],
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
    iris_target_name,
    n_jobs: int,
) -> None:
    iris_classifier_crossfit_dual_target = (
        iris_classifier_ranker_dual_target.best_model_crossfit_
    )

    with pytest.raises(
        ValueError,
        match=(
            f"only single-output classifiers .* are supported.*"
            f"{iris_target_name}.*{iris_target_name}2"
        ),
    ):
        LearnerInspector(n_jobs=n_jobs).fit(
            crossfit=iris_classifier_crossfit_dual_target
        )


def test_shap_plot_data(
    iris_sample,
    iris_inspector_multi_class: LearnerInspector[ClassifierPipelineDF],
) -> None:
    shap_plot_data = iris_inspector_multi_class.shap_plot_data()
    # noinspection SpellCheckingInspection
    assert tuple(iris_inspector_multi_class.output_names_) == (
        "setosa",
        "versicolor",
        "virginica",
    )

    features_shape = shap_plot_data.features.shape
    shap_values = shap_plot_data.shap_values
    assert isinstance(shap_values, list)
    assert len(shap_values) == 3
    assert all(isinstance(shap, np.ndarray) for shap in shap_values)
    assert all(shap.shape == features_shape for shap in shap_values)

    shap_index = shap_plot_data.features.index
    assert_frame_equal(shap_plot_data.features, iris_sample.features.loc[shap_index])
    assert_series_equal(shap_plot_data.target, iris_sample.target.loc[shap_index])


def _fit_learner_ranker(
    sample: Sample, cv: BaseCrossValidator, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    # define parameters and crossfit
    grids = [
        LearnerGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(random_state=42), preprocessing=None
            ),
            learner_parameters={"n_estimators": [10, 50], "min_samples_leaf": [4, 8]},
        )
    ]
    # pipeline inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    return LearnerRanker(
        grids=grids,
        cv=cv,
        scoring="f1_macro",
        # shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=sample)


def call_inspector_method_both_algorithms(
    method: Callable[..., T], **kwargs: Any
) -> Tuple[T, T]:
    # noinspection PyUnresolvedReferences
    inspector: LearnerInspector = method.__self__
    legacy = inspector._legacy
    try:
        inspector._legacy = True
        legacy_result = method(**kwargs)
        inspector._legacy = False
        result = method(**kwargs)
    finally:
        inspector._legacy = legacy

    return result, legacy_result
