import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.utils import Bunch

from gamma.ml import Sample
from gamma.ml.selection import ModelParameterGrid
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression import (
    AdaBoostRegressorDF,
    DecisionTreeRegressorDF,
    ExtraTreeRegressorDF,
    LGBMRegressorDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
    SVRDF,
)
from test import read_test_config
from test.gamma.ml import make_simple_transformer
from test.paths import TEST_DATA_CSV

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# disable SHAP debugging messages
logging.getLogger("shap").setLevel(logging.WARNING)

warnings.filterwarnings(
    "ignore", message=r"Starting from version 2", category=UserWarning
)


@pytest.fixture
def inputfile_config() -> Dict[str, Any]:
    return read_test_config(section="inputfile")


@pytest.fixture
def boston_target() -> str:
    return "target"


@pytest.fixture
def iris_target() -> str:
    return "target"


@pytest.fixture
def n_jobs() -> int:
    return -3


@pytest.fixture
def batch_table(inputfile_config: Dict[str, Any]) -> pd.DataFrame:

    return pd.read_csv(
        filepath_or_buffer=TEST_DATA_CSV,
        delimiter=inputfile_config["delimiter"],
        header=inputfile_config["header"],
        decimal=inputfile_config["decimal"],
    )


@pytest.fixture
def regressor_grids(simple_preprocessor) -> List[ModelParameterGrid]:
    RANDOM_STATE = {f"random_state": [42]}
    return [
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=LGBMRegressorDF()
            ),
            estimator_parameters={
                "max_depth": [5, 10],
                "min_split_gain": [0.1, 0.2],
                "num_leaves": [50, 100, 200],
                **RANDOM_STATE,
            },
        ),
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=AdaBoostRegressorDF()
            ),
            estimator_parameters={"n_estimators": [50, 80], **RANDOM_STATE},
        ),
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=RandomForestRegressorDF()
            ),
            estimator_parameters={"n_estimators": [50, 80], **RANDOM_STATE},
        ),
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=DecisionTreeRegressorDF()
            ),
            estimator_parameters={
                "max_depth": [0.5, 1.0],
                "max_features": [0.5, 1.0],
                **RANDOM_STATE,
            },
        ),
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=ExtraTreeRegressorDF()
            ),
            estimator_parameters={"max_depth": [5, 10, 12], **RANDOM_STATE},
        ),
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=SVRDF()
            ),
            estimator_parameters={"gamma": [0.5, 1], "C": [50, 100]},
        ),
        ModelParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=LinearRegressionDF()
            ),
            estimator_parameters={"normalize": [False, True]},
        ),
    ]


@pytest.fixture
def sample(batch_table: pd.DataFrame, inputfile_config: Dict[str, Any]) -> Sample:
    # drop columns that should not take part in model
    batch_table = batch_table.drop(columns=["Date", "Batch Id"])

    # replace values of +/- infinite with n/a, then drop all n/a columns:
    batch_table = batch_table.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="all"
    )

    sample = Sample(
        observations=batch_table, target_name=inputfile_config["yield_column_name"]
    )
    return sample


@pytest.fixture
def simple_preprocessor(sample: Sample) -> TransformerDF:
    return make_simple_transformer(
        impute_median_columns=sample.features_by_type(Sample.DTYPE_NUMERICAL).columns,
        one_hot_encode_columns=sample.features_by_type(Sample.DTYPE_OBJECT).columns,
    )


@pytest.fixture
def boston_df(boston_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

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

    iris_df = pd.DataFrame(
        data=np.c_[iris.data, iris.target], columns=[*iris.feature_names, iris_target]
    )

    # replace target numericals with actual class labels
    iris_df.loc[:, iris_target] = iris_df.loc[:, iris_target].apply(
        lambda x: iris.target_names[int(x)]
    )

    return iris_df


@pytest.fixture
def iris_sample(iris_df: pd.DataFrame, iris_target: str) -> Sample:
    return Sample(observations=iris_df, target_name=iris_target)
