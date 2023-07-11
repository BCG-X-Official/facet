"""
Model inspector tests.
"""
import logging
import platform
import warnings
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas._testing import assert_index_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pytools.data import LinkageTree, Matrix
from pytools.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from sklearndf import ClassifierDF, __sklearn_1_1__, __sklearn_version__
from sklearndf.classification import (
    GradientBoostingClassifierDF,
    RandomForestClassifierDF,
)
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from ..conftest import check_ranking
from facet.data import Sample
from facet.explanation import (
    ExactExplainerFactory,
    KernelExplainerFactory,
    PermutationExplainerFactory,
    TreeExplainerFactory,
)
from facet.explanation.base import ExplainerFactory
from facet.inspection import FunctionInspector, LearnerInspector, NativeLearnerInspector
from facet.selection import LearnerSelector

# noinspection PyMissingOrEmptyDocstring

log = logging.getLogger(__name__)

T = TypeVar("T")


def test_regressor_selector(
    regressor_selector: LearnerSelector[
        RegressorPipelineDF[LGBMRegressorDF], GridSearchCV
    ]
) -> None:
    scores_expected: List[float] = (
        [0.578, 0.530, 0.310, 0.308, 0.294, 0.226, 0.217, 0.217, 0.217, 0.217]
        if (
            __sklearn_version__ < __sklearn_1_1__
            or platform.machine() != "arm64"
            or platform.system() != "Darwin"
        )
        # on M1 macs, we get different results starting with scikit-learn 1.1
        else [0.579, 0.531, 0.311, 0.308, 0.246, 0.217, 0.217, 0.217, 0.217, 0.217]
    )
    check_ranking(
        ranking=regressor_selector.summary_report(),
        is_classifier=False,
        scores_expected=scores_expected,
        params_expected=None,
    )


@pytest.mark.parametrize(  # type: ignore
    argnames=("explainer_factory_cls", "explainer_factory_args"),
    argvalues=[
        (
            TreeExplainerFactory,
            dict(
                feature_perturbation="tree_path_dependent", uses_background_dataset=True
            ),
        ),
        (KernelExplainerFactory, dict(link="identity", data_size_limit=8)),
        (ExactExplainerFactory, {}),
        (PermutationExplainerFactory, {}),
    ],
)
@pytest.mark.parametrize(  # type: ignore
    argnames="native",
    argvalues=(False, True),
)
def test_model_inspection(
    explainer_factory_cls: Type[ExplainerFactory[LGBMRegressorDF]],
    explainer_factory_args: Dict[str, Any],
    best_lgbm_model: RegressorPipelineDF[LGBMRegressorDF],
    sample: Sample,
    n_jobs: int,
    native: bool,
) -> None:
    # test the ModelInspector with the given explainer factory:

    explainer_factory: ExplainerFactory[LGBMRegressorDF] = explainer_factory_cls(
        **explainer_factory_args
    )

    inspector: Union[
        LearnerInspector[RegressorPipelineDF[LGBMRegressorDF]],
        NativeLearnerInspector[Pipeline],
    ]

    regressor_feature_names: Set[str]  # column index names

    if native:
        assert (
            best_lgbm_model.preprocessing is not None
        ), "preprocessing step must be defined"

        regressor = best_lgbm_model.regressor.native_estimator

        if __sklearn_version__ < __sklearn_1_1__:
            # scikit-learn 1.0.x does not support output feature names in simple
            # imputers, so we cannot use this for preprocessing
            log.warning(
                f"scikit-learn {__sklearn_version__} does not support output "
                "feature names in simple imputers, so we will test the native learner "
                "inspector without preprocessing"
            )
            assert (
                sample.features.notna().all().all()
            ), "observations must not contain missing values"
            model = regressor
            regressor_feature_names = set(sample.feature_names)

        else:
            # scikit-learn 1.1.x supports output feature names in simple imputers,
            # so we can use this for preprocessing
            model = Pipeline(
                # create a native pipeline from the regressor pipeline
                steps=[
                    (
                        "preprocessing",
                        best_lgbm_model.preprocessing.native_estimator,
                    ),
                    ("regressor", regressor),
                ]
            ).fit(X=sample.features, y=sample.target)
            regressor_feature_names = set(model[:-1].get_feature_names_out())

        # noinspection PyTypeChecker
        inspector = NativeLearnerInspector(
            model=model,
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
        ).fit(sample)
    else:
        inspector = LearnerInspector(
            model=best_lgbm_model,
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
        ).fit(sample)
        regressor_feature_names = set(best_lgbm_model.regressor.feature_names_in_)

    shap_values: pd.DataFrame = inspector.shap_values()

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values) == len(sample)

    # index names
    assert shap_values.index.names == [Sample.IDX_OBSERVATION]
    assert shap_values.columns.names == [Sample.IDX_FEATURE]

    # check that the column names are the same as the feature names
    assert set(shap_values.columns) == set(regressor_feature_names)

    # check that the row order has been preserved
    assert_index_equal(shap_values.index, sample.index)

    # check that the SHAP values add up to the predictions
    shap_totals = shap_values.sum(axis=1)

    # calculate the difference between total SHAP values and prediction
    # for every observation. This is always the same constant value,
    # therefore the mean absolute deviation is zero.

    shap_minus_pred = shap_totals - inspector.model.predict(X=sample.features)
    assert (
        round((shap_minus_pred - shap_minus_pred.mean()).abs().mean(), 12) == 0.0
    ), "predictions matching total SHAP"

    # validate the linkage tree of the resulting inspector

    # if the inspector supports interaction values, test the redundancy linkage
    # otherwise test the association linkage
    if inspector.shap_interaction:
        linkage = inspector.feature_redundancy_linkage()
        mode = "Redundancy"
    else:
        linkage = inspector.feature_association_linkage()
        mode = "Association"

    # validate the linkage tree

    assert isinstance(linkage, LinkageTree)
    # get the node whose child distance is < 0.8, and confirm it is the only one
    cluster_nodes = [
        node
        for node in linkage.iter_nodes()
        if not node.is_leaf and node.children_distance < 0.7
    ]
    assert len(cluster_nodes) == 1, "only two features form a cluster"
    # check the child nodes are Longitude and Latitude
    children = linkage.children(cluster_nodes[0])
    assert children is not None, "a cluster node has children"
    assert {child.name.split("__")[-1] for child in children} == (
        {"Longitude", "Latitude"}
    ), "the cluster is Longitude and Latitude features"

    print()
    DendrogramDrawer(style="text").draw(
        data=linkage,
        title=f"{inspector.explainer_factory.__class__.__name__} ({mode})",
    )


def test_binary_classifier_ranking(
    iris_classifier_selector_binary: LearnerSelector[
        ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV
    ]
) -> None:
    expected_learner_scores = [0.938, 0.936, 0.936, 0.929]

    ranking = iris_classifier_selector_binary.summary_report()

    log.debug(f"\n{ranking}")

    check_ranking(
        ranking=ranking,
        is_classifier=True,
        scores_expected=expected_learner_scores,
        params_expected={
            2: dict(min_samples_leaf=4, n_estimators=10),
            3: dict(min_samples_leaf=8, n_estimators=10),
        },
    )


# noinspection DuplicatedCode
def test_model_inspection_classifier_binary(
    iris_classifier_binary: ClassifierPipelineDF[RandomForestClassifierDF],
    iris_sample_binary: Sample,
    n_jobs: int,
) -> None:
    model_inspector = LearnerInspector(
        model=iris_classifier_binary,
        shap_interaction=False,
        n_jobs=n_jobs,
    ).fit(iris_sample_binary)

    # calculate the shap value matrix, without any consolidation
    shap_values = model_inspector.shap_values()

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, model=iris_classifier_binary, sample=iris_sample_binary
    )

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values) == len(iris_sample_binary)

    # Shap decomposition matrices (feature dependencies)

    try:
        association_matrix = cast(
            Matrix[np.float_],
            model_inspector.feature_association_matrix(
                clustered=True, symmetrical=True
            ),
        )
        assert_allclose(
            association_matrix.values,
            np.array(
                [
                    [np.nan, 0.684, 0.368, 0.002],
                    [0.684, np.nan, 0.442, 0.000],
                    [0.368, 0.442, np.nan, 0.010],
                    [0.002, 0.000, 0.010, np.nan],
                ]
            ),
            atol=0.02,
        )
    except AssertionError as error:
        print_expected_matrix(error=error)
        raise

    linkage_tree = cast(LinkageTree, model_inspector.feature_association_linkage())

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


def test_model_inspection_classifier_binary_single_shap_output(n_jobs: int) -> None:
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

    # fit the model
    pipeline = ClassifierPipelineDF(
        classifier=GradientBoostingClassifierDF(random_state=42)
    ).fit(sample_df.features, sample_df.target)

    # fit the inspector
    LearnerInspector(model=pipeline, n_jobs=n_jobs).fit(sample_df)


# noinspection DuplicatedCode
def test_model_inspection_classifier_multi_class(
    iris_inspector_multi_class: LearnerInspector[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
) -> None:
    iris_classifier = iris_inspector_multi_class.model
    iris_sample = iris_inspector_multi_class.sample_

    # calculate the shap value matrix, without any consolidation
    shap_values = iris_inspector_multi_class.shap_values()

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, model=iris_classifier, sample=iris_sample
    )

    # Feature importance

    feature_importance: pd.DataFrame = iris_inspector_multi_class.feature_importance()
    assert feature_importance.index.equals(
        pd.Index(iris_sample.feature_names, name="feature")
    )
    assert feature_importance.columns.equals(
        pd.Index(iris_inspector_multi_class.output_names, name="class")
    )
    assert_allclose(
        feature_importance.values,
        np.array(
            [
                [0.122, 0.086, 0.102],
                [0.020, 0.021, 0.007],
                [0.433, 0.465, 0.481],
                [0.424, 0.428, 0.410],
            ]
        ),
        atol=0.02,
    )

    # Shap decomposition matrices (feature dependencies)

    try:
        synergy_matrix = cast(
            List[Matrix[np.float_]],
            iris_inspector_multi_class.feature_synergy_matrix(clustered=False),
        )

        assert_allclose(
            np.hstack([m.values for m in synergy_matrix]),
            np.array(
                [
                    [np.nan, 0.008, 0.032, 0.037, np.nan, 0.002]
                    + [0.367, 0.343, np.nan, 0.001, 0.081, 0.067],
                    [0.124, np.nan, 0.042, 0.035, 0.094, np.nan]
                    + [0.061, 0.055, 0.160, np.nan, 0.643, 0.456],
                    [0.002, 0.000, np.nan, 0.003, 0.041, 0.008]
                    + [np.nan, 0.048, 0.015, 0.000, np.nan, 0.034],
                    [0.002, 0.000, 0.003, np.nan, 0.025, 0.009]
                    + [0.042, np.nan, 0.008, 0.012, 0.034, np.nan],
                ]
            ),
            atol=0.02,
        )

        redundancy_matrix = cast(
            List[Matrix[np.float_]],
            iris_inspector_multi_class.feature_redundancy_matrix(clustered=False),
        )
        assert_allclose(
            np.hstack([m.values for m in redundancy_matrix]),
            np.array(
                [
                    [np.nan, 0.080, 0.734, 0.721, np.nan, 0.156]
                    + [0.327, 0.315, np.nan, 0.002, 0.671, 0.610],
                    [0.071, np.nan, 0.382, 0.388, 0.142, np.nan]
                    + [0.333, 0.403, 0.002, np.nan, 0.039, 0.021],
                    [0.757, 0.398, np.nan, 0.995, 0.495, 0.352]
                    + [np.nan, 0.741, 0.720, 0.109, np.nan, 0.754],
                    [0.747, 0.402, 0.995, np.nan, 0.468, 0.423]
                    + [0.746, np.nan, 0.649, 0.038, 0.753, np.nan],
                ]
            ),
            atol=0.02,
        )

        association_matrix = cast(
            List[Matrix[np.float_]],
            iris_inspector_multi_class.feature_association_matrix(clustered=False),
        )
        assert_allclose(
            np.hstack([m.values for m in association_matrix]),
            np.array(
                [
                    [np.nan, 0.087, 0.746, 0.735, np.nan, 0.132]
                    + [0.466, 0.419, np.nan, 0.003, 0.719, 0.643],
                    [0.087, np.nan, 0.387, 0.390, 0.132, np.nan]
                    + [0.357, 0.428, 0.003, np.nan, 0.034, 0.046],
                    [0.746, 0.387, np.nan, 0.998, 0.466, 0.357]
                    + [np.nan, 0.788, 0.719, 0.034, np.nan, 0.787],
                    [0.735, 0.390, 0.998, np.nan, 0.419, 0.428]
                    + [0.788, np.nan, 0.643, 0.046, 0.787, np.nan],
                ]
            ),
            atol=0.02,
        )
    except AssertionError as error:
        print_expected_matrix(error=error, split=True)
        raise

    linkage_trees = cast(
        List[LinkageTree], iris_inspector_multi_class.feature_association_linkage()
    )

    for output, linkage_tree in zip(
        iris_inspector_multi_class.output_names, linkage_trees
    ):
        print()
        DendrogramDrawer(style=DendrogramReportStyle()).draw(
            data=linkage_tree, title=f"Iris feature association linkage: {output}"
        )


def _validate_shap_values_against_predictions(
    shap_values: pd.DataFrame, model: ClassifierDF, sample: Sample
) -> None:
    # calculate the matching predictions, so we can check if the SHAP values add up
    # correctly
    predicted_probabilities: pd.DataFrame = model.predict_proba(sample.features)

    assert isinstance(
        predicted_probabilities, pd.DataFrame
    ), "predicted probabilities are single-output"

    expected_probability_range = 1 / len(predicted_probabilities.columns)

    def _check_probabilities(
        _class_probabilities: pd.DataFrame, _shap_for_split_and_class: pd.Series
    ) -> None:
        expected_probability = _class_probabilities.join(_shap_for_split_and_class).sum(
            axis=1
        )

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
        # for binary classification we have SHAP values only for the second class
        _check_probabilities(
            predicted_probabilities.iloc[:, [1]],
            -shap_values.sum(axis=1).rename("shap"),
        )

    else:
        # multi-class classification has outputs for each class

        for class_idx, class_name in enumerate(predicted_probabilities.columns):
            # for each observation and class, we expect to get the constant
            # expected probability value by deducting the SHAP values for all
            # features from the predicted probability

            class_probabilities = predicted_probabilities.loc[:, [class_name]]

            shap_for_split_and_class = (
                -shap_values[class_idx].sum(axis=1).rename("shap")
            )

            _check_probabilities(class_probabilities, shap_for_split_and_class)


# noinspection DuplicatedCode
@pytest.mark.parametrize(  # type: ignore
    argnames="native",
    argvalues=(False, True),
)
def test_model_inspection_classifier_interaction(
    iris_classifier_binary: ClassifierPipelineDF[RandomForestClassifierDF],
    iris_sample_binary: Sample,
    n_jobs: int,
    native: bool,
) -> None:
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    cls_inspector: Type[
        Union[
            LearnerInspector[RandomForestClassifierDF],
            NativeLearnerInspector[RandomForestClassifier],
        ]
    ]
    learner: Union[RandomForestClassifierDF, RandomForestClassifier]
    if native:
        cls_inspector = NativeLearnerInspector[RandomForestClassifier]
        learner = iris_classifier_binary.final_estimator.native_estimator
    else:
        cls_inspector = LearnerInspector[RandomForestClassifierDF]
        learner = iris_classifier_binary.final_estimator

    model_inspector = cls_inspector(
        model=learner,
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", uses_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(iris_sample_binary)

    model_inspector_no_interaction = cls_inspector(
        model=learner,
        shap_interaction=False,
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", uses_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(iris_sample_binary)

    # calculate shap interaction values
    shap_interaction_values: pd.DataFrame = model_inspector.shap_interaction_values()

    # calculate shap values from interaction values
    shap_values = shap_interaction_values.groupby(by="observation").sum()

    # shap interaction values add up to shap values
    # we have to live with differences of up to 0.020, given the different results
    # returned for SHAP values and SHAP interaction values
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
        shap_values=shap_interaction_values.groupby(level=0).sum(),
        model=iris_classifier_binary,
        sample=iris_sample_binary,
    )

    assert model_inspector.feature_importance().values == pytest.approx(
        np.array([0.054, 0.019, 0.451, 0.477]), abs=0.02
    )

    try:
        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_synergy_matrix(
                    clustered=False, symmetrical=True
                ),
            ).values,
            np.array(
                [
                    [np.nan, 0.011, 0.006, 0.007],
                    [0.011, np.nan, 0.006, 0.007],
                    [0.006, 0.006, np.nan, 0.003],
                    [0.007, 0.007, 0.003, np.nan],
                ]
            ),
            atol=0.02,
        )
        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_synergy_matrix(absolute=True, symmetrical=True),
            ).values,
            np.array(
                [
                    [np.nan, 0.001, 0.002, 0.001],
                    [0.001, np.nan, 0.000, 0.002],
                    [0.002, 0.000, np.nan, 0.002],
                    [0.001, 0.002, 0.002, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_synergy_matrix(clustered=True),
            ).values,
            np.array(
                [
                    [np.nan, 0.000, 0.000, 0.001],
                    [0.386, np.nan, 0.108, 0.314],
                    [0.005, 0.002, np.nan, 0.059],
                    [0.002, 0.000, 0.001, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_], model_inspector.feature_synergy_matrix(absolute=True)
            ).values,
            np.array(
                [
                    [np.nan, 0.000, 0.000, 0.001],
                    [0.003, np.nan, 0.001, 0.003],
                    [0.003, 0.000, np.nan, 0.003],
                    [0.001, 0.000, 0.001, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_redundancy_matrix(
                    clustered=False, symmetrical=True
                ),
            ).values,
            np.array(
                [
                    [np.nan, 0.013, 0.462, 0.383],
                    [0.013, np.nan, 0.000, 0.003],
                    [0.462, 0.000, np.nan, 0.677],
                    [0.383, 0.003, 0.677, np.nan],
                ]
            ),
            atol=0.02,
        )
        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_redundancy_matrix(
                    absolute=True, symmetrical=True
                ),
            ).values,
            np.array(
                [
                    [np.nan, 0.314, 0.102, 0.001],
                    [0.314, np.nan, 0.116, 0.000],
                    [0.102, 0.116, np.nan, 0.000],
                    [0.001, 0.000, 0.000, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_redundancy_matrix(clustered=True),
            ).values,
            np.array(
                [
                    [np.nan, 0.677, 0.384, 0.003],
                    [0.676, np.nan, 0.465, 0.000],
                    [0.382, 0.438, np.nan, 0.013],
                    [0.002, 0.000, 0.012, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_redundancy_matrix(absolute=True),
            ).values,
            np.array(
                [
                    [np.nan, 0.323, 0.183, 0.002],
                    [0.305, np.nan, 0.209, 0.000],
                    [0.021, 0.024, np.nan, 0.001],
                    [0.000, 0.000, 0.000, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_association_matrix(
                    clustered=False, symmetrical=True
                ),
            ).values,
            np.array(
                [
                    [np.nan, 0.009, 0.447, 0.383],
                    [0.009, np.nan, 0.000, 0.001],
                    [0.447, 0.000, np.nan, 0.678],
                    [0.383, 0.001, 0.678, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_association_matrix(
                    absolute=True, symmetrical=True
                ),
            ).values,
            np.array(
                [
                    [np.nan, 0.314, 0.102, 0.000],
                    [0.314, np.nan, 0.113, 0.000],
                    [0.102, 0.113, np.nan, 0.000],
                    [0.000, 0.000, 0.000, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_association_matrix(clustered=True),
            ).values,
            np.array(
                [
                    [np.nan, 0.678, 0.383, 0.001],
                    [0.678, np.nan, 0.447, 0.000],
                    [0.383, 0.447, np.nan, 0.009],
                    [0.001, 0.000, 0.009, np.nan],
                ]
            ),
            atol=0.02,
        )

        assert_allclose(
            cast(
                Matrix[np.float_],
                model_inspector.feature_association_matrix(absolute=True),
            ).values,
            np.array(
                [
                    [np.nan, 0.323, 0.182, 0.001],
                    [0.305, np.nan, 0.201, 0.000],
                    [0.021, 0.024, np.nan, 0.000],
                    [0.000, 0.000, 0.000, np.nan],
                ]
            ),
            atol=0.02,
        )

    except AssertionError as error:
        print_expected_matrix(error=error)
        raise

    linkage_tree = cast(LinkageTree, model_inspector.feature_redundancy_linkage())

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature redundancy linkage"
    )


def test_model_inspection_classifier_interaction_dual_target(
    iris_sample_binary_dual_target: Sample,
    iris_classifier_selector_dual_target: LearnerSelector[
        ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV
    ],
    iris_target_name: str,
    n_jobs: int,
) -> None:
    iris_classifier_dual_target = iris_classifier_selector_dual_target.best_estimator_

    with pytest.raises(
        ValueError,
        match=(
            f"only single-target classifiers .* are supported.*"
            f"{iris_target_name}.*{iris_target_name}2"
        ),
    ):
        LearnerInspector(model=iris_classifier_dual_target, n_jobs=n_jobs).fit(
            iris_sample_binary_dual_target
        )


def test_shap_plot_data(
    iris_sample_multi_class: Sample,
    iris_inspector_multi_class: LearnerInspector[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
) -> None:
    shap_plot_data = iris_inspector_multi_class.shap_plot_data()
    # noinspection SpellCheckingInspection
    assert tuple(iris_inspector_multi_class.output_names) == (
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
    assert_frame_equal(
        shap_plot_data.features, iris_sample_multi_class.features.loc[shap_index]
    )
    assert_series_equal(
        shap_plot_data.target, iris_sample_multi_class.target.loc[shap_index]
    )


def test_function_inspector(n_jobs: int) -> None:
    # define a function to inspect, taking a 2D array with 3 columns as input,
    # calculating (x1 + x2) * x3

    pi2 = 2 * np.pi

    def model_function(x: pd.DataFrame) -> pd.Series:
        return np.sin(pi2 * x.x1) * np.sin(pi2 * (x.x2 + x.x3) / 2.0) + x.x4 + x.x5

    # create a background dataset, with 1000 random samples
    observations = pd.DataFrame(
        np.random.random(size=(1000, 4)), columns=["x1", "x2", "x4", "x5"]
    )
    # column x3 is the same as x2
    observations["x3"] = observations["x2"]
    # add a column with the target values
    observations["y"] = model_function(observations)
    # create a sample from the background dataset
    background = Sample(
        observations=observations,
        target_name="y",
    )

    # create a function inspector
    inspector = FunctionInspector(
        model=model_function,
        feature_names=background.feature_names,
        explainer_factory=ExactExplainerFactory(),
        n_jobs=1,
    )

    # fit the inspector
    inspector.fit(background)

    # print the redundancy and synergy linkage using dendrogram drawers
    print()
    DendrogramDrawer(style="text").draw(
        data=cast(LinkageTree, inspector.feature_redundancy_linkage()),
        title="FunctionInspector (Redundancy)",
    )
    DendrogramDrawer(style="text").draw(
        data=cast(LinkageTree, inspector.feature_synergy_linkage()),
        title="FunctionInspector (Synergy)",
    )


#
# Utility functions
#


def print_expected_matrix(error: AssertionError, *, split: bool = False) -> None:
    # print expected output for copy/paste into assertion statement

    import re

    array: Optional[re.Match[str]] = re.search(r"array\(([^)]+)\)", error.args[0])
    if array is not None:
        matrix: List[List[float]] = eval(
            array[1].replace(r"\n", "\n").replace("nan", "np.nan")
        )

        print_matrix(matrix, split=split)


def print_matrix(matrix: List[List[float]], *, split: bool) -> None:
    print("==== matrix assertion failed ====\nExpected Matrix:")
    print("[")
    for row in matrix:
        txt = "    ["
        halfpoint = len(row) // 2
        for i, x in enumerate(row):
            if split and i == halfpoint:
                txt += "] + ["
            elif i > 0:
                txt += ", "
            txt += "np.nan" if np.isnan(x) else f"{x:.3f}"
        print(txt + "],")
    print("]")
