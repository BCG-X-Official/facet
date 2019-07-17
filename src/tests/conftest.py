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
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils import Bunch

from tests import read_test_config
from tests.model import make_simple_transformer
from tests.paths import TEST_DATA_CSV
from yieldengine import Sample
from yieldengine.df.transform import DataFrameTransformer
from yieldengine.model import RegressionModel
from yieldengine.model.selection import ModelGrid

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# disable SHAP debugging messages
logging.getLogger("shap").setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore", message=r"Starting from version 2", category=UserWarning
)


@pytest.fixture
def boston_target() -> str:
    return "target"


@pytest.fixture
def iris_target() -> str:
    return "target"


@pytest.fixture
def available_cpus() -> int:
    return -3


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
def regressor_grids(simple_preprocessor) -> List[ModelGrid]:
    RANDOM_STATE = {f"random_state": [42]}
    return [
        ModelGrid(
            model=RegressionModel(
                preprocessing=simple_preprocessor, estimator=LGBMRegressor()
            ),
            estimator_parameters={
                "max_depth": [5, 10],
                "min_split_gain": [0.1, 0.2],
                "num_leaves": [50, 100, 200],
                **RANDOM_STATE,
            },
        ),
        ModelGrid(
            model=RegressionModel(
                preprocessing=simple_preprocessor, estimator=AdaBoostRegressor()
            ),
            estimator_parameters={"n_estimators": [50, 80], **RANDOM_STATE},
        ),
        ModelGrid(
            model=RegressionModel(
                preprocessing=simple_preprocessor, estimator=RandomForestRegressor()
            ),
            estimator_parameters={"n_estimators": [50, 80], **RANDOM_STATE},
        ),
        ModelGrid(
            model=RegressionModel(
                preprocessing=simple_preprocessor, estimator=DecisionTreeRegressor()
            ),
            estimator_parameters={
                "max_depth": [0.5, 1.0],
                "max_features": [0.5, 1.0],
                **RANDOM_STATE,
            },
        ),
        ModelGrid(
            model=RegressionModel(
                preprocessing=simple_preprocessor, estimator=ExtraTreeRegressor()
            ),
            estimator_parameters={"max_depth": [5, 10, 12], **RANDOM_STATE},
        ),
        ModelGrid(
            model=RegressionModel(preprocessing=simple_preprocessor, estimator=SVR()),
            estimator_parameters={"gamma": [0.5, 1], "C": [50, 100]},
        ),
        ModelGrid(
            model=RegressionModel(
                preprocessing=simple_preprocessor, estimator=LinearRegression()
            ),
            estimator_parameters={"normalize": [False, True]},
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
def simple_preprocessor(sample: Sample) -> DataFrameTransformer:
    return make_simple_transformer(
        impute_median_columns=sample.features_by_type(Sample.DTYPE_NUMERICAL).columns,
        one_hot_encode_columns=sample.features_by_type(Sample.DTYPE_OBJECT).columns,
    )


@pytest.fixture
def boston_df(boston_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

    # use first 100 rows only, since KernelExplainer is very slow...
    return pd.DataFrame(
        data=np.c_[boston.data, boston.target],
        columns=[*boston.feature_names, boston_target],
    )


@pytest.fixture
def boston_sample(boston_df: pd.DataFrame, boston_target: str) -> Sample:
    return Sample(observations=boston_df, target_name=boston_target)


@pytest.fixture
def iris_df(iris_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    iris: Bunch = datasets.load_iris()

    # use first 100 rows only, since KernelExplainer is very slow...
    return pd.DataFrame(
        data=np.c_[iris.data, iris.target], columns=[*iris.feature_names, iris_target]
    )


@pytest.fixture
def iris_sample(iris_df: pd.DataFrame, iris_target: str) -> Sample:
    return Sample(observations=iris_df, target_name=iris_target)
