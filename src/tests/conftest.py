import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
import pytest
from lightgbm.sklearn import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from tests import read_test_config
from tests.paths import TEST_DATA_CSV
from yieldengine import Sample
from yieldengine.model.selection import Model
from yieldengine.pipeline import (
    make_simple_transformer_step,
    ModelPipeline,
    TransformationStep,
)

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings(
    "ignore", message=r"Starting from version 2", category=UserWarning
)


@pytest.fixture
def batch_table() -> pd.DataFrame:

    # Note: this file is not included within the git repository!
    inputfile_config = read_test_config(section="inputfile")
    return pd.read_csv(
        filepath_or_buffer=TEST_DATA_CSV,
        delimiter=inputfile_config["delimiter"],
        header=inputfile_config["header"],
        decimal=inputfile_config["decimal"],
    )


@pytest.fixture
def regressor_grids(transformer_step: TransformationStep) -> List[Model]:
    RANDOM_STATE = {f"{ModelPipeline.STEP_MODEL}__random_state": [42]}
    return [
        Model(
            pipeline=ModelPipeline(
                preprocessing=[transformer_step], estimator=LGBMRegressor()
            ),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__max_depth": [5, 10],
                f"{ModelPipeline.STEP_MODEL}__min_split_gain": [0.1, 0.2],
                f"{ModelPipeline.STEP_MODEL}__num_leaves": [50, 100, 200],
                **RANDOM_STATE,
            },
        ),
        Model(
            pipeline=ModelPipeline(
                preprocessing=[transformer_step], estimator=AdaBoostRegressor()
            ),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__n_estimators": [50, 80],
                **RANDOM_STATE,
            },
        ),
        Model(
            pipeline=ModelPipeline(
                preprocessing=[transformer_step], estimator=RandomForestRegressor()
            ),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__n_estimators": [50, 80],
                **RANDOM_STATE,
            },
        ),
        Model(
            pipeline=ModelPipeline(
                preprocessing=[transformer_step], estimator=DecisionTreeRegressor()
            ),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__max_depth": [0.5, 1.0],
                f"{ModelPipeline.STEP_MODEL}__max_features": [0.5, 1.0],
                **RANDOM_STATE,
            },
        ),
        Model(
            pipeline=ModelPipeline(
                preprocessing=[transformer_step], estimator=ExtraTreeRegressor()
            ),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__max_depth": [5, 10, 12],
                **RANDOM_STATE,
            },
        ),
        Model(
            pipeline=ModelPipeline(preprocessing=[transformer_step], estimator=SVR()),
            parameter_grid={
                f"{ModelPipeline.STEP_MODEL}__gamma": [0.5, 1],
                f"{ModelPipeline.STEP_MODEL}__C": [50, 100],
            },
        ),
        Model(
            pipeline=ModelPipeline(
                preprocessing=[transformer_step], estimator=LinearRegression()
            ),
            parameter_grid={f"{ModelPipeline.STEP_MODEL}__normalize": [False, True]},
        ),
    ]


@pytest.fixture
def sample(batch_table: pd.DataFrame) -> Sample:
    # drop columns that should not take part in model
    batch_table = batch_table.drop(columns=["Date", "Batch Id"])

    # replace values of +/- infinite with n/a, then drop all n/a columns:
    batch_table = batch_table.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="all"
    )

    sample = Sample(observations=batch_table, target_name="Yield")
    return sample


@pytest.fixture
def transformer_step(sample: Sample) -> TransformationStep:

    return make_simple_transformer_step(
        impute_median=sample.features_by_type(Sample.DTYPE_NUMERICAL).columns,
        one_hot_encode=sample.features_by_type(Sample.DTYPE_OBJECT).columns,
    )
