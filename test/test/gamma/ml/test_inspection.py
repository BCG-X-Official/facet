"""
Model inspector tests.
"""
import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
import pytest
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
from gamma.sklearndf.regression import (
    AdaBoostRegressorDF,
    ExtraTreeRegressorDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
)
from gamma.sklearndf.regression.extra import LGBMRegressorDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from test.gamma.ml import check_ranking

log = logging.getLogger(__name__)


# noinspection PyMissingOrEmptyDocstring


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
        expected_regressors = [
            AdaBoostRegressorDF,
            AdaBoostRegressorDF,
            RandomForestRegressorDF,
            RandomForestRegressorDF,
            LinearRegressionDF,
            LinearRegressionDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
        ]
        expected_parameters = {
            0: dict(regressor__n_estimators=80, regressor__random_state=42),
            1: dict(regressor__n_estimators=50, regressor__random_state=42),
            2: dict(regressor__n_estimators=80, regressor__random_state=42),
            3: dict(regressor__n_estimators=50, regressor__random_state=42),
        }
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
        expected_regressors = [
            AdaBoostRegressorDF,
            AdaBoostRegressorDF,
            RandomForestRegressorDF,
            RandomForestRegressorDF,
            ExtraTreeRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
            LGBMRegressorDF,
        ]
        expected_parameters = {
            0: dict(regressor__n_estimators=50, regressor__random_state=42),
            1: dict(regressor__n_estimators=80, regressor__random_state=42),
            2: dict(regressor__n_estimators=80, regressor__random_state=42),
            3: dict(regressor__n_estimators=50, regressor__random_state=42),
        }

    log.debug(f"\n{regressor_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=regressor_ranker.ranking(),
        expected_scores=expected_scores,
        expected_learners=expected_regressors,
        expected_parameters=expected_parameters,
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
            return TreeExplainer(
                model=estimator, feature_dependence="independent", data=data
            )
        except Exception as e:
            log.debug(
                f"failed to instantiate shap.TreeExplainer:{str(e)},"
                "using shap.KernelExplainer as fallback"
            )
            # noinspection PyUnresolvedReferences
            return KernelExplainer(model=estimator.predict, data=data)

    # noinspection PyTypeChecker
    inspector_2 = RegressorInspector(explainer_factory=_ef, shap_interaction=False).fit(
        crossfit=best_lgbm_crossfit
    )
    inspector_2.shap_values()

    linkage_tree = inspector_2.feature_association_linkage()

    print()
    DendrogramDrawer(style="text").draw(data=linkage_tree, title="Test")


def test_model_inspection_classifier(
    iris_sample: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define checksums for this test
    checksum_shap = 16811895885973514240
    checksum_association_matrix = 2376971889351401631
    expected_learner_scores = [1.0, 1.0, 0.946, 0.891]

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
    test_sample: Sample = iris_sample.subsample(
        loc=iris_sample.target.isin(iris_sample.target.unique()[:2])
    )

    model_ranker: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ] = LearnerRanker(
        grid=models,
        cv=cv_bootstrap,
        scoring="f1_macro",
        # shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(
        sample=test_sample
    )

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking(),
        expected_scores=expected_learner_scores,
        expected_learners=[RandomForestClassifierDF] * 4,
        expected_parameters={
            2: dict(classifier__min_samples_leaf=32, classifier__n_estimators=50),
            3: dict(classifier__min_samples_leaf=32, classifier__n_estimators=10),
        },
    )

    crossfit = model_ranker.best_model_crossfit

    model_inspector = ClassifierInspector(shap_interaction=False).fit(crossfit=crossfit)
    # make and check shap value matrix
    shap_matrix = model_inspector.shap_values()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values)
        == checksum_shap
    )

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix) == len(test_sample)

    # Shap decomposition matrices (feature dependencies)
    feature_associations: pd.DataFrame = model_inspector.feature_association_matrix()
    log.info(feature_associations)
    # check number of rows
    assert len(feature_associations) == len(test_sample.feature_columns)
    assert len(feature_associations.columns) == len(test_sample.feature_columns)

    # check association values
    for c in feature_associations.columns:
        fa = feature_associations.loc[:, c]
        assert 0.0 <= fa.min() <= fa.max() <= 1.0

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(feature_associations.round(decimals=4)).values)
        == checksum_association_matrix
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Test"
    )
