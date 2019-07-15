import logging
import warnings
from typing import *

import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold
from sklearn.svm import SVR

from yieldengine import Sample
from yieldengine.df.transform import DataFrameTransformer
from yieldengine.model import (
    ClassificationModel,
    ProbabilityCalibrationMethod,
    RegressionModel,
)
from yieldengine.model.inspection import ModelInspector
from yieldengine.model.prediction import ModelFitCV
from yieldengine.model.selection import (
    ModelEvaluation,
    ModelGrid,
    ModelRanker,
    summary_report,
)
from yieldengine.model.validation import CircularCrossValidator

log = logging.getLogger(__name__)

K_FOLDS: int = 5
TEST_RATIO = 1 / K_FOLDS
N_SPLITS = K_FOLDS * 2


def test_model_inspection(available_cpus: int, boston_sample: Sample) -> None:
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
        ModelGrid(
            model=RegressionModel(estimator=SVR(gamma="scale"), preprocessing=None),
            estimator_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        ),
        ModelGrid(
            model=RegressionModel(estimator=LGBMRegressor(), preprocessing=None),
            estimator_parameters={
                "max_depth": (1, 2, 5),
                "min_split_gain": (0.1, 0.2, 0.5),
                "num_leaves": (2, 3),
            },
        ),
    ]

    # use first 100 rows only, since KernelExplainer is very slow...

    test_sample: Sample = boston_sample.select_observations(numbers=range(0, 100))

    model_ranker: ModelRanker = ModelRanker(
        grids=models, cv=test_cv, scoring="neg_mean_squared_error"
    )

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=available_cpus
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    best_svr = [m for m in model_ranking if isinstance(m.model.estimator, SVR)][0]
    best_lgbm = [
        model_evaluation
        for model_evaluation in model_ranking
        if isinstance(model_evaluation.model.estimator, LGBMRegressor)
    ][0]

    for model_evaluation in best_svr, best_lgbm:
        model_fit = ModelFitCV(
            model=model_evaluation.model,
            cv=test_cv,
            sample=test_sample,
            n_jobs=available_cpus,
        )

        # test predictions_for_all_samples
        predictions_df: pd.DataFrame = model_fit.predictions_for_all_splits()
        assert ModelFitCV.F_PREDICTION in predictions_df.columns
        assert ModelFitCV.F_TARGET in predictions_df.columns

        # check number of split ids
        assert (
            predictions_df.index.get_level_values(level=ModelFitCV.F_SPLIT_ID).nunique()
            == N_SPLITS
        )

        # check correct number of rows
        allowed_variance = 0.01
        assert (
            (len(test_sample) * (TEST_RATIO - allowed_variance) * N_SPLITS)
            <= len(predictions_df)
            <= (len(test_sample) * (TEST_RATIO + allowed_variance) * N_SPLITS)
        )

        model_inspector = ModelInspector(model_fit=model_fit)
        # make and check shap value matrix
        shap_matrix = model_inspector.shap_matrix()

        # the length of rows in shap_matrix should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(shap_matrix) == len(test_sample)

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = model_inspector.feature_dependency_matrix()
        log.info(corr_matrix)
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


def test_model_inspection_with_encoding(
    batch_table: pd.DataFrame,
    regressor_grids: Iterable[ModelGrid],
    sample: Sample,
    simple_preprocessor: DataFrameTransformer,
    available_cpus: int,
) -> None:

    # define the circular cross validator with just 5 splits (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_splits=5)

    model_ranker: ModelRanker = ModelRanker(
        grids=regressor_grids, cv=circular_cv, scoring="r2"
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        sample=sample, n_jobs=available_cpus
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    # best_svr = [m for m in model_ranking if isinstance(m.estimator, SVR)][0]
    best_lgbm = [
        model_evaluation
        for model_evaluation in model_ranking
        if isinstance(model_evaluation.model.estimator, LGBMRegressor)
    ][0]
    for model_evaluation in [best_lgbm]:
        model_fit = ModelFitCV(
            model=model_evaluation.model,
            cv=circular_cv,
            sample=sample,
            n_jobs=available_cpus,
        )
        mi = ModelInspector(model_fit=model_fit)

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

        mi2 = ModelInspector(model_fit=model_fit, explainer_factory=ef)
        mi2.shap_matrix()


def test_model_inspection_classifier(available_cpus: int, iris_sample: Sample) -> None:
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
        ModelGrid(
            model=ClassificationModel(
                estimator=RandomForestClassifier(),
                preprocessing=None,
                calibration=ProbabilityCalibrationMethod.NO_CALIBRATION,
            ),
            estimator_parameters={"n_estimators": [50, 80]},
        )
    ]

    # adjust iris-sample to only include classes 0 and 1 - since (for now) only
    # binary classification is supported
    test_sample: Sample = iris_sample.select_observations(
        numbers=iris_sample.index[iris_sample.target.isin([0, 1])]
    )

    model_ranker: ModelRanker = ModelRanker(
        grids=models, cv=test_cv, scoring="f1_macro"
    )

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=available_cpus
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    model_evaluation = model_ranking[0]

    model_fit = ModelFitCV(
        model=model_evaluation.model,
        cv=test_cv,
        sample=test_sample,
        n_jobs=available_cpus,
    )

    # test predictions_for_all_samples
    predictions_df: pd.DataFrame = model_fit.predictions_for_all_splits()
    assert ModelFitCV.F_PREDICTION in predictions_df.columns
    assert ModelFitCV.F_TARGET in predictions_df.columns

    # check number of split ids
    assert (
        predictions_df.index.get_level_values(level=ModelFitCV.F_SPLIT_ID).nunique()
        == N_SPLITS
    )

    # check correct number of rows
    allowed_variance = 0.01
    assert (
        (len(test_sample) * (TEST_RATIO - allowed_variance) * N_SPLITS)
        <= len(predictions_df)
        <= (len(test_sample) * (TEST_RATIO + allowed_variance) * N_SPLITS)
    )

    model_inspector = ModelInspector(model_fit=model_fit)
    # make and check shap value matrix
    shap_matrix = model_inspector.shap_matrix()

    # the length of rows in shap_matrix should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix) == len(test_sample)

    # correlated shap matrix: feature dependencies
    corr_matrix: pd.DataFrame = model_inspector.feature_dependency_matrix()
    log.info(corr_matrix)
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
