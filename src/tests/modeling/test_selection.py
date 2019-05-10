import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
import pytest
from lightgbm.sklearn import LGBMRegressor
from sklearn import datasets
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

# noinspection PyUnresolvedReferences
from tests.shared_fixtures import batch_table
from yieldengine.loading.sample import Sample
from yieldengine.modeling.factory import (
    ModelPipelineFactory,
    SimplePreprocessingPipelineFactory,
)
from yieldengine.modeling.selection import Model, ModelRanker, ModelRanking, RankedModel
from yieldengine.modeling.validation import CircularCrossValidator

log = logging.getLogger(__name__)


@pytest.fixture
def models() -> List[Model]:
    return [
        Model(
            estimator=LGBMRegressor(),
            parameter_grid={
                "max_depth": (5, 10),
                "min_split_gain": (0.1, 0.2),
                "num_leaves": (50, 100, 200),
            },
        ),
        Model(estimator=AdaBoostRegressor(), parameter_grid={"n_estimators": (50, 80)}),
        Model(
            estimator=RandomForestRegressor(), parameter_grid={"n_estimators": (50, 80)}
        ),
        Model(
            estimator=DecisionTreeRegressor(),
            parameter_grid={"max_depth": (0.5, 1.0), "max_features": (0.5, 1.0)},
        ),
        Model(
            estimator=ExtraTreeRegressor(), parameter_grid={"max_depth": (5, 10, 12)}
        ),
        Model(estimator=SVR(), parameter_grid={"gamma": (0.5, 1), "C": (50, 100)}),
        Model(
            estimator=LinearRegression(), parameter_grid={"normalize": (False, True)}
        ),
    ]


@pytest.fixture
def sample(batch_table: pd.DataFrame) -> Sample:
    # drop columns that should not take part in modeling
    batch_table = batch_table.drop(columns=["Date", "Batch Id"])

    # replace values of +/- infinite with n/a, then drop all n/a columns:
    batch_table = batch_table.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="all"
    )

    sample = Sample(observations=batch_table, target_name="Yield")
    return sample


@pytest.fixture
def preprocessing_pipeline_factory(sample: Sample) -> ModelPipelineFactory:
    # define a ColumnTransformer to pre-process:
    # define a sklearn Pipeline step, containing the preprocessor defined above:
    return SimplePreprocessingPipelineFactory(
        impute_mean=sample.features_by_type(dtype=sample.DTYPE_NUMERICAL),
        one_hot_encode=sample.features_by_type(dtype=sample.DTYPE_OBJECT),
    )


def test_model_ranker(
    batch_table: pd.DataFrame,
    models: List[Model],
    sample: Sample,
    preprocessing_pipeline_factory: ModelPipelineFactory,
) -> None:
    # define the circular cross validator with just 5 folds (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_folds=5)

    model_ranker: ModelRanker = ModelRanker(
        models=models,
        pipeline_factory=preprocessing_pipeline_factory,
        cv=circular_cv,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: ModelRanking = model_ranker.run(sample=sample)

    assert len(model_ranking) > 0
    assert isinstance(
        model_ranking.model(rank=ModelRanking.BEST_MODEL_RANK), RankedModel
    )
    assert (
        model_ranking.model(rank=0).ranking_score
        >= model_ranking.model(rank=1).ranking_score
        >= model_ranking.model(rank=2).ranking_score
        >= model_ranking.model(rank=3).ranking_score
        >= model_ranking.model(rank=4).ranking_score
        >= model_ranking.model(rank=len(model_ranking) - 1).ranking_score
    )

    # check if parameters set for estimators actually match expected:
    for r in range(0, len(model_ranking)):
        m: RankedModel = model_ranking.model(r)
        assert set(m.parameters).issubset(m.estimator.get_params())

    log.info(f"\n{model_ranking}")


def test_model_ranker_no_preprocessing() -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a yield-engine circular CV:
    cv = CircularCrossValidator(test_ratio=0.21, num_folds=50)

    # define parameters and model
    models = [
        Model(
            estimator=SVC(gamma="scale"),
            parameter_grid={"kernel": ("linear", "rbf"), "C": [1, 10]},
        )
    ]

    model_ranker: ModelRanker = ModelRanker(models=models, cv=cv)

    #  load sklearn test-data and convert to pd
    iris = datasets.load_iris()
    test_data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    test_sample: Sample = Sample(observations=test_data, target_name="target")

    model_ranking: ModelRanking = model_ranker.run(test_sample)

    log.info(f"\n{model_ranking.summary_report()}")

    assert (
        model_ranking.model(ModelRanking.BEST_MODEL_RANK).ranking_score >= 0.8
    ), "Expected a performance of at least 0.8"
