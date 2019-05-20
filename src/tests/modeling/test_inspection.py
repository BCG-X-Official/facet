import logging
import warnings

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn import datasets
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from sklearn.utils import Bunch

from yieldengine import Sample
from yieldengine.model.inspection import ModelInspector
from yieldengine.model.selection import Model, ModelRanker, ModelRanking
from yieldengine.model.validation import CircularCrossValidator
from yieldengine.pipeline import ModelPipeline

log = logging.getLogger(__name__)


def test_model_inspection() -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    N_FOLDS = 5
    TEST_RATIO = 0.2
    BOSTON_TARGET = "target"

    # define a yield-engine circular CV:
    test_cv = ShuffleSplit(n_splits=N_FOLDS, test_size=TEST_RATIO, random_state=42)

    # define parameters and models
    models = [
        Model(
            pipeline=ModelPipeline(estimator=SVR(gamma="scale"), preprocessing=None),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__kernel": ("linear", "rbf"),
                f"{ModelPipeline.STEP_MODEL}__C": [1, 10],
            },
        ),
        Model(
            pipeline=ModelPipeline(estimator=LGBMRegressor(), preprocessing=None),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__max_depth": (1, 2, 5),
                f"{ModelPipeline.STEP_MODEL}__min_split_gain": (0.1, 0.2, 0.5),
                f"{ModelPipeline.STEP_MODEL}__num_leaves": (2, 3),
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
        models=models, cv=test_cv, scoring="neg_mean_squared_error"
    )

    model_ranking: ModelRanking = model_ranker.run(test_sample)

    # consider: model_with_type(...) function for ModelRanking
    best_svr = [m for m in model_ranking if isinstance(m.pipeline.estimator, SVR)][0]
    best_lgbm = [
        m for m in model_ranking if isinstance(m.pipeline.estimator, LGBMRegressor)
    ][0]

    for ranked_model in best_svr, best_lgbm:

        mi = ModelInspector(
            pipeline=ranked_model.pipeline, cv=test_cv, sample=test_sample
        )

        # test predictions_for_all_samples
        predictions_df: pd.DataFrame = mi.predictions_for_all_samples()

        assert ModelInspector.F_FOLD_ID in predictions_df.columns
        assert ModelInspector.F_PREDICTION in predictions_df.columns

        # check number of fold-starts
        assert predictions_df[ModelInspector.F_FOLD_ID].nunique() == N_FOLDS

        # check correct number of rows
        ALLOWED_VARIANCE = 0.01
        assert (
            (len(test_sample) * (TEST_RATIO - ALLOWED_VARIANCE) * N_FOLDS)
            <= len(predictions_df)
            <= (len(test_sample) * (TEST_RATIO + ALLOWED_VARIANCE) * N_FOLDS)
        )

        # make and check shap value matrix
        shap_matrix = mi.shap_matrix()

        # the length of rows in shap_matrix should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(shap_matrix) == len(predictions_df.index.unique())

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = mi.feature_dependencies()

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

        estimator, clustered_corr_matrix = mi.run_clustering_on_feature_correlations()


def test_model_inspection_with_encoding(
    batch_table: pd.DataFrame, regressor_grids, sample: Sample, transformer_step
) -> None:

    # define the circular cross validator with just 5 folds (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_folds=5)

    model_ranker: ModelRanker = ModelRanker(
        models=regressor_grids, cv=circular_cv, scoring="r2"
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: ModelRanking = model_ranker.run(sample=sample)

    log.info(model_ranking)

    # consider: model_with_type(...) function for ModelRanking
    # best_svr = [m for m in model_ranking if isinstance(m.estimator, SVR)][0]
    best_lgbm = [
        m for m in model_ranking if isinstance(m.pipeline.estimator, LGBMRegressor)
    ][0]
    for model in [best_lgbm]:
        mi = ModelInspector(pipeline=model.pipeline, cv=circular_cv, sample=sample)

        shap_matrix = mi.shap_matrix()

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = mi.feature_dependencies()

        # cluster feature importances
        estimator, clustered_corr_matrix = mi.run_clustering_on_feature_correlations()
