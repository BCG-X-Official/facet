"""
Model inspector tests.
"""
import functools
import logging
import operator
import warnings
from typing import *

import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_pandas_object
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import ClassifierInspector, RegressorInspector
from gamma.ml.selection import ClassifierRanker, ParameterGrid, RegressorRanker
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from test.gamma.ml import check_ranking

log = logging.getLogger(__name__)


# noinspection PyMissingOrEmptyDocstring


def test_model_inspection(
    batch_table: pd.DataFrame,
    regressor_grids: Sequence[ParameterGrid],
    regressor_ranker: RegressorRanker,
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF],
    regressor_inspector: RegressorInspector,
    cv: BaseCrossValidator,
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs: int,
    fast_execution: bool,
) -> None:
    if fast_execution:
        # define checksums for this test
        checksum_shap = 16212594514483871543
        checksum_association_matrix = 8969182771326710805

        checksum_learner_scores = -3.723311
        checksum_learner_ranks = "6a3cfb540e56298cdccebc5c72dae7aa"
    else:
        # define checksums for this test
        checksum_shap = 4647247706471413882
        checksum_association_matrix = 7913166565570533555

        checksum_learner_scores = -7.939242
        checksum_learner_ranks = "5e4b373d56a53647c9483a5606235c9a"

    log.debug(f"\n{regressor_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=regressor_ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    shap_values = regressor_inspector.shap_values()

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values) == len(sample)

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_values.round(decimals=4)).values)
        == checksum_shap
    )

    # correlated shap matrix: feature dependencies
    association_matrix: pd.DataFrame = regressor_inspector.feature_association_matrix()

    # determine number of unique features across the models in the crossfit
    n_features = len(
        functools.reduce(
            operator.or_,
            (set(model.features_out) for model in best_lgbm_crossfit.models()),
        )
    )

    # check that dimensions of pairwise feature matrices are equal to # of features
    for matrix, matrix_name in zip(
        (
            association_matrix,
            regressor_inspector.feature_synergy_matrix(),
            regressor_inspector.feature_redundancy_matrix(),
        ),
        ("association", "synergy", "redundancy"),
    ):
        matrix_full_name = f"feature {matrix_name} matrix"
        assert len(matrix) == n_features, f"rows in {matrix_full_name}"
        assert len(matrix.columns) == n_features, f"columns in {matrix_full_name}"

    # check correlation values
    for c in association_matrix.columns:
        assert (
            -1.0
            <= association_matrix.fillna(0).loc[:, c].min()
            <= association_matrix.fillna(0).loc[:, c].max()
            <= 1.0
        )

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(association_matrix.round(decimals=4)).values)
        == checksum_association_matrix
    )

    # cluster associated features
    _linkage = regressor_inspector.feature_association_linkage()

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
    iris_sample: Sample, cv: BaseCrossValidator, n_jobs: int
) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define checksums for this test
    checksum_shap = 5207601201651574496
    checksum_association_matrix = 5535519327633455357
    checksum_learner_scores = 2.0
    checksum_learner_ranks = "a8fe61f0f98c078fbcf427ad344c1749"

    # define parameters and crossfit
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(), preprocessing=None
            ),
            learner_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    # pipeline inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    test_sample: Sample = iris_sample.subsample(
        loc=iris_sample.target.isin(iris_sample.target.unique()[0:2])
    )

    model_ranker = ClassifierRanker(
        grid=models,
        cv=cv,
        scoring="f1_macro",
        shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=test_sample)

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
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

    # correlated shap matrix: feature dependencies
    corr_matrix: pd.DataFrame = model_inspector.feature_association_matrix()
    log.info(corr_matrix)
    # check number of rows
    assert len(corr_matrix) == len(test_sample.feature_columns)
    assert len(corr_matrix.columns) == len(test_sample.feature_columns)

    # check correlation values
    for c in corr_matrix.columns:
        c_corr = corr_matrix.loc[:, c]
        assert -1.0 <= c_corr.min() <= c_corr.max() <= 1.0

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(corr_matrix.round(decimals=4)).values)
        == checksum_association_matrix
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Test"
    )
