"""
Model inspector tests.
"""
import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_pandas_object
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import (
    ClassifierInspector,
    kernel_explainer_factory,
    RegressorInspector,
    tree_explainer_factory,
)
from gamma.ml.selection import ClassifierRanker, ParameterGrid, RegressorRanker
from gamma.ml.validation import CircularCV
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from gamma.sklearndf.regression import SVRDF
from gamma.sklearndf.regression.extra import LGBMRegressorDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from test.gamma.ml import check_ranking

log = logging.getLogger(__name__)

K_FOLDS: int = 5
TEST_RATIO = 1 / K_FOLDS
N_SPLITS = K_FOLDS * 2


def test_model_inspection(n_jobs, boston_sample: Sample) -> None:
    # checksums for the model inspection test - one for the LGBM, one for the SVR
    checksums_shap = (17573313757033027070, 13285572916961982080)
    checksum_corr_matrix = (13028973179387096991, 18397646897559448061)
    checksum_learner_scores = -218.87516793944133
    checksum_learner_ranks = "0972fa60fd9beb2c1f8be21324506f4d"

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameter grid
    grid = [
        ParameterGrid(
            pipeline=(
                RegressorPipelineDF(regressor=SVRDF(gamma="scale"), preprocessing=None)
            ),
            learner_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                regressor=LGBMRegressorDF(), preprocessing=None
            ),
            learner_parameters={
                "max_depth": (1, 2, 5),
                "min_split_gain": (0.1, 0.2, 0.5),
                "num_leaves": (2, 3),
            },
        ),
    ]

    # use first 100 rows only, since KernelExplainer is very slow...

    test_sample: Sample = boston_sample.subsample(iloc=slice(100))

    ranker = RegressorRanker(
        grid=grid, cv=test_cv, scoring="neg_mean_squared_error", n_jobs=n_jobs
    ).fit(sample=test_sample)

    log.debug(f"\n{ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    ranking = ranker.ranking()

    # consider: model_with_type(...) function for ModelRanking
    best_svr = [
        model for model in ranking if isinstance(model.pipeline.regressor, SVRDF)
    ][0]
    best_lgbm = [
        model_evaluation
        for model_evaluation in ranking
        if isinstance(model_evaluation.pipeline.regressor, LGBMRegressorDF)
    ][0]

    for model_index, (model_evaluation, factory) in enumerate(
        ((best_lgbm, tree_explainer_factory), (best_svr, kernel_explainer_factory))
    ):

        pipeline: RegressorPipelineDF = model_evaluation.pipeline
        model_fit = LearnerCrossfit(pipeline=pipeline, cv=test_cv, random_state=42).fit(
            sample=test_sample
        )

        model_inspector = RegressorInspector(
            crossfit=model_fit, explainer_factory=factory
        )
        # make and check shap value matrix
        shap_matrix = model_inspector.shap_values()

        # the length of rows in shap_values should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(shap_matrix) == len(test_sample)

        # check actual values using checksum:
        #
        assert (
            np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values)
            == checksums_shap[model_index]
        )

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = model_inspector.feature_association_matrix()

        # check number of rows
        assert len(corr_matrix) == len(test_sample.feature_columns)
        assert len(corr_matrix.columns) == len(test_sample.feature_columns)

        # check correlation values
        for c in corr_matrix.columns:
            assert (
                -1.0
                <= corr_matrix.fillna(0).loc[:, c].min()
                <= corr_matrix.fillna(0).loc[:, c].max()
                <= 1.0
            )

        # check actual values using checksum:
        assert (
            np.sum(hash_pandas_object(corr_matrix.round(decimals=4)).values)
            == checksum_corr_matrix[model_index]
        )

        linkage_tree = model_inspector.feature_association_linkage()

        DendrogramDrawer(style=DendrogramReportStyle()).draw(
            data=linkage_tree, title="Test"
        )


def test_model_inspection_with_encoding(
    batch_table: pd.DataFrame,
    regressor_grids: Sequence[ParameterGrid],
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs,
) -> None:
    # define checksums for this test
    checksum_shap = 3533151003136279125
    checksum_corr_matrix = 2280660998565980993

    checksum_learner_scores = -7.8631
    checksum_learner_ranks = "2d763e35c03b309994f6c8585cacb035"

    # define the circular cross validator with just 5 splits (to speed up testing)
    cv = CircularCV(test_ratio=0.20, n_splits=5)

    ranker: RegressorRanker = RegressorRanker(
        grid=regressor_grids, cv=cv, scoring="r2", n_jobs=n_jobs
    ).fit(sample=sample)

    log.debug(f"\n{ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    # we get the best model_evaluation which is a LGBM - for the sake of test
    # performance
    validation = [
        validation
        for validation in ranker.ranking()
        if isinstance(validation.pipeline.regressor, LGBMRegressorDF)
    ][0]

    validation_model = LearnerCrossfit(
        pipeline=validation.pipeline, cv=cv, random_state=42, n_jobs=n_jobs
    ).fit(sample=sample)

    mi = RegressorInspector(crossfit=validation_model)

    shap_matrix = mi.shap_values()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values)
        == checksum_shap
    )

    # correlated shap matrix: feature dependencies
    corr_matrix: pd.DataFrame = mi.feature_association_matrix()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(corr_matrix.round(decimals=4)).values)
        == checksum_corr_matrix
    )

    # cluster feature importances
    _linkage = mi.feature_association_linkage()

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

    mi2 = RegressorInspector(crossfit=validation_model, explainer_factory=_ef)
    mi2.shap_values()

    linkage_tree = mi2.feature_association_linkage()
    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Test"
    )


def test_model_inspection_classifier(n_jobs, iris_sample: Sample) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define checksums for this test
    checksum_shap = 6629520117757454166
    checksum_corr_matrix = 11698715255607208353
    checksum_learner_scores = 2.0
    checksum_learner_ranks = "a8fe61f0f98c078fbcf427ad344c1749"

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

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
        grid=models, cv=test_cv, scoring="f1_macro", n_jobs=n_jobs
    ).fit(sample=test_sample)

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    crossfit = model_ranker.best_model_crossfit(random_state=42)

    model_inspector = ClassifierInspector(crossfit=crossfit)
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
        == checksum_corr_matrix
    )

    linkage_tree = model_inspector.feature_association_linkage()
    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Test"
    )
