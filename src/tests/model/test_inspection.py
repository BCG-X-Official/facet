import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn import datasets
from sklearn.model_selection import BaseCrossValidator, ShuffleSplit
from sklearn.svm import SVR
from sklearn.utils import Bunch

from yieldengine import Sample
from yieldengine.df.transform import DataFrameTransformer
from yieldengine.model import Model
from yieldengine.model.inspection import ModelInspector
from yieldengine.model.prediction import PredictorCV
from yieldengine.model.selection import (
    ModelEvaluation,
    ModelGrid,
    ModelRanker,
    summary_report,
)
from yieldengine.model.validation import CircularCrossValidator

log = logging.getLogger(__name__)

N_SPLITS = 5
TEST_RATIO = 0.2
BOSTON_TARGET = "target"


def test_model_inspection(available_cpus: int) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = ShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_RATIO, random_state=42
    )

    # define parameters and models
    models = [
        ModelGrid(
            model=Model(estimator=SVR(gamma="scale"), preprocessing=None),
            estimator_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        ),
        ModelGrid(
            model=Model(estimator=LGBMRegressor(), preprocessing=None),
            estimator_parameters={
                "max_depth": (1, 2, 5),
                "min_split_gain": (0.1, 0.2, 0.5),
                "num_leaves": (2, 3),
            },
        ),
    ]

    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

    # use first 100 rows only, since KernelExplainer is very slow...
    test_data = pd.DataFrame(
        data=np.c_[boston.data, boston.target],
        columns=[*boston.feature_names, BOSTON_TARGET],
    ).loc[:100,]

    test_sample: Sample = Sample(observations=test_data, target_name=BOSTON_TARGET)

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
        mp = PredictorCV(model=model_evaluation.model, cv=test_cv, sample=test_sample)

        # test predictions_for_all_samples
        predictions_df: pd.DataFrame = mp.predictions_for_all_samples()

        assert PredictorCV.F_SPLIT_ID in predictions_df.columns
        assert PredictorCV.F_PREDICTION in predictions_df.columns

        # check number of split starts
        assert predictions_df[PredictorCV.F_SPLIT_ID].nunique() == N_SPLITS

        # check correct number of rows
        ALLOWED_VARIANCE = 0.01
        assert (
            (len(test_sample) * (TEST_RATIO - ALLOWED_VARIANCE) * N_SPLITS)
            <= len(predictions_df)
            <= (len(test_sample) * (TEST_RATIO + ALLOWED_VARIANCE) * N_SPLITS)
        )

        mi = ModelInspector(predictor=mp)
        # make and check shap value matrix
        shap_matrix = mi.shap_matrix()

        # the length of rows in shap_matrix should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(shap_matrix) == len(predictions_df.index.unique())

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = mi.feature_dependency_matrix()

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

        linkage_tree = mi.cluster_dependent_features()


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
        mp = PredictorCV(model=model_evaluation.model, cv=circular_cv, sample=sample)
        mi = ModelInspector(predictor=mp)

        shap_matrix = mi.shap_matrix()

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = mi.feature_dependency_matrix()

        # cluster feature importances
        linkage_tree = mi.cluster_dependent_features()
