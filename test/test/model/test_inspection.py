import logging
import warnings
from typing import *

import pandas as pd
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

from gamma.ml import Sample
from gamma.ml.fitcv import ClassifierFitCV, RegressorFitCV
from gamma.ml.inspection import ClassificationModelInspector, RegressionModelInspector
from gamma.ml.selection import (
    ModelEvaluation,
    ModelParameterGrid,
    ModelRanker,
    summary_report,
)
from gamma.ml.validation import CircularCrossValidator
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassificationPipelineDF, RegressionPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF, SVRDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle

log = logging.getLogger(__name__)

K_FOLDS: int = 5
TEST_RATIO = 1 / K_FOLDS
N_SPLITS = K_FOLDS * 2


def test_model_inspection(n_jobs, boston_sample: Sample) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameters and models
    models = [
        ModelParameterGrid(
            pipeline=(
                RegressionPipelineDF(regressor=SVRDF(gamma="scale"), preprocessing=None)
            ),
            estimator_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        ),
        ModelParameterGrid(
            pipeline=RegressionPipelineDF(
                regressor=LGBMRegressorDF(), preprocessing=None
            ),
            estimator_parameters={
                "max_depth": (1, 2, 5),
                "min_split_gain": (0.1, 0.2, 0.5),
                "num_leaves": (2, 3),
            },
        ),
    ]

    # use first 100 rows only, since KernelExplainer is very slow...

    test_sample: Sample = boston_sample.select_observations_by_position(
        positions=range(0, 100)
    )

    model_ranker: ModelRanker = ModelRanker(
        grids=models, cv=test_cv, scoring="neg_mean_squared_error"
    )

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=n_jobs
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    best_svr = [m for m in model_ranking if isinstance(m.model.regressor, SVRDF)][0]
    best_lgbm = [
        model_evaluation
        for model_evaluation in model_ranking
        if isinstance(model_evaluation.model.regressor, LGBMRegressorDF)
    ][0]

    for model_evaluation in best_svr, best_lgbm:
        model_fit = RegressorFitCV(
            pipeline=model_evaluation.model, cv=test_cv, sample=test_sample
        )

        # test predictions_for_all_samples
        predictions_df: pd.DataFrame = model_fit.predictions_for_all_splits()
        assert RegressorFitCV.F_PREDICTION in predictions_df.columns
        assert RegressorFitCV.F_TARGET in predictions_df.columns

        # check number of split ids
        assert (
            predictions_df.index.get_level_values(
                level=RegressorFitCV.F_SPLIT_ID
            ).nunique()
            == N_SPLITS
        )

        # check correct number of rows
        allowed_variance = 0.01
        assert (
            (len(test_sample) * (TEST_RATIO - allowed_variance) * N_SPLITS)
            <= len(predictions_df)
            <= (len(test_sample) * (TEST_RATIO + allowed_variance) * N_SPLITS)
        )

        model_inspector = RegressionModelInspector(models=model_fit)
        # make and check shap value matrix
        shap_matrix = model_inspector.shap_matrix()

        # the length of rows in shap_matrix should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(shap_matrix) == len(test_sample)

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = model_inspector.feature_dependency_matrix()
        # check number of rows
        assert len(corr_matrix) == len(test_sample.feature_names) - 1
        assert len(corr_matrix.columns) == len(test_sample.feature_names) - 1

        # check correlation values
        for c in corr_matrix.columns:
            assert (
                -1.0
                <= corr_matrix.fillna(0).loc[:, c].min()
                <= corr_matrix.fillna(0).loc[:, c].max()
                <= 1.0
            )

        linkage_tree = model_inspector.cluster_dependent_features()
        print()
        DendrogramDrawer(
            title="Test", linkage_tree=linkage_tree, style=DendrogramReportStyle()
        ).draw()


def test_model_inspection_with_encoding(
    batch_table: pd.DataFrame,
    regressor_grids: Sequence[ModelParameterGrid],
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs,
) -> None:

    # define the circular cross validator with just 5 splits (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_splits=5)

    model_ranker: ModelRanker = ModelRanker(
        grids=regressor_grids, cv=circular_cv, scoring="r2"
    )

    model = regressor_grids[0].pipeline

    # run the ModelRanker to retrieve a ranking
    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        sample=sample, n_jobs=n_jobs
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    # best_svr = [m for m in model_ranking if isinstance(m.estimator, SVR)][0]
    best_lgbm = [
        model_evaluation
        for model_evaluation in model_ranking
        if isinstance(model_evaluation.model.regressor, LGBMRegressorDF)
    ][0]
    for model_evaluation in [best_lgbm]:
        model_fit = RegressorFitCV(
            pipeline=model_evaluation.model, cv=circular_cv, sample=sample
        )
        mi = RegressionModelInspector(models=model_fit)

        shap_matrix = mi.shap_matrix()

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = mi.feature_dependency_matrix()

        # cluster feature importances
        linkage_tree = mi.cluster_dependent_features()

        #  test the ModelInspector with a custom ExplainerFactory:
        def ef(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:

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

        mi2 = RegressionModelInspector(models=model_fit, explainer_factory=ef)
        mi2.shap_matrix()

        linkage_tree = mi2.cluster_dependent_features()
        print()
        DendrogramDrawer(
            title="Test", linkage_tree=linkage_tree, style=DendrogramReportStyle()
        ).draw()


def test_model_inspection_classifier(n_jobs, iris_sample: Sample) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameters and models
    models = [
        ModelParameterGrid(
            pipeline=ClassificationPipelineDF(
                classifier=RandomForestClassifierDF(), preprocessing=None
            ),
            estimator_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    # model inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    test_sample: Sample = iris_sample.select_observations_by_index(
        ids=iris_sample.target.isin(iris_sample.target.unique()[0:2])
    )

    model_ranker: ModelRanker = ModelRanker(
        grids=models, cv=test_cv, scoring="f1_macro"
    )

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=n_jobs
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    model_evaluation = model_ranking[0]

    model_fit = ClassifierFitCV(
        pipeline=model_evaluation.model,
        cv=test_cv,
        sample=test_sample,
        calibration=ClassifierFitCV.SIGMOID,
        n_jobs=n_jobs,
    )

    model_inspector = ClassificationModelInspector(models=model_fit)
    # make and check shap value matrix
    shap_matrix = model_inspector.shap_matrix()

    # the length of rows in shap_matrix should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix) == len(test_sample)

    # correlated shap matrix: feature dependencies
    corr_matrix: pd.DataFrame = model_inspector.feature_dependency_matrix()
    log.info(corr_matrix)
    # check number of rows
    assert len(corr_matrix) == len(test_sample.feature_names)
    assert len(corr_matrix.columns) == len(test_sample.feature_names)

    # check correlation values
    for c in corr_matrix.columns:
        assert (
            -1.0
            <= corr_matrix.fillna(0).loc[:, c].min()
            <= corr_matrix.fillna(0).loc[:, c].max()
            <= 1.0
        )

    linkage_tree = model_inspector.cluster_dependent_features()
    print()
    DendrogramDrawer(
        title="Test", linkage_tree=linkage_tree, style=DendrogramReportStyle()
    ).draw()
