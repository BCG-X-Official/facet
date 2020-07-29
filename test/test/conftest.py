import functools
import logging
import operator
import os
from typing import *

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils import Bunch

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import RegressorInspector
from gamma.ml.selection import LearnerEvaluation, LearnerRanker, ParameterGrid
from gamma.ml.validation import BootstrapCV
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression import (
    AdaBoostRegressorDF,
    DecisionTreeRegressorDF,
    ExtraTreeRegressorDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
    SVRDF,
)
from gamma.sklearndf.regression.extra import LGBMRegressorDF
from test.gamma.ml import make_simple_transformer

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# disable SHAP debugging messages
logging.getLogger("shap").setLevel(logging.WARNING)


K_FOLDS = 5
N_BOOTSTRAPS = 30


@pytest.fixture
def boston_target() -> str:
    return "target"


@pytest.fixture
def iris_target() -> str:
    return "target"


@pytest.fixture
def n_jobs() -> int:
    return -1


@pytest.fixture
def fast_execution() -> bool:
    return os.environ.get("FAST_EXECUTION", "1") == "1"


@pytest.fixture
def cv_kfold() -> KFold:
    # define a CV
    return KFold(n_splits=K_FOLDS)


@pytest.fixture
def cv_bootstrap() -> BaseCrossValidator:
    # define a CV
    return BootstrapCV(n_splits=N_BOOTSTRAPS, random_state=42)


@pytest.fixture
def regressor_grids(simple_preprocessor: TransformerDF) -> List[ParameterGrid]:
    random_state = {f"random_state": [42]}

    return [
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=LGBMRegressorDF()
            ),
            learner_parameters={
                "max_depth": [5, 10],
                "min_split_gain": [0.1, 0.2],
                "num_leaves": [50, 100, 200],
                **random_state,
            },
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=AdaBoostRegressorDF()
            ),
            learner_parameters={"n_estimators": [50, 80], **random_state},
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=RandomForestRegressorDF()
            ),
            learner_parameters={"n_estimators": [50, 80], **random_state},
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=DecisionTreeRegressorDF()
            ),
            learner_parameters={
                "max_depth": [0.5, 1.0],
                "max_features": [0.5, 1.0],
                **random_state,
            },
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=ExtraTreeRegressorDF()
            ),
            learner_parameters={"max_depth": [5, 10, 12], **random_state},
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=SVRDF()
            ),
            learner_parameters={"gamma": [0.5, 1], "C": [50, 100]},
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=LinearRegressionDF()
            ),
            learner_parameters={"normalize": [False, True]},
        ),
    ]


@pytest.fixture
def regressor_ranker(
    cv_kfold: KFold,
    regressor_grids: List[ParameterGrid[RegressorPipelineDF]],
    sample: Sample,
    n_jobs: int,
) -> LearnerRanker[RegressorPipelineDF]:
    return LearnerRanker(
        grid=regressor_grids, cv=cv_kfold, scoring="r2", n_jobs=n_jobs
    ).fit(sample=sample)


@pytest.fixture
def best_lgbm_crossfit(
    regressor_ranker: LearnerRanker[RegressorPipelineDF],
    cv_kfold: KFold,
    sample: Sample,
    n_jobs: int,
) -> LearnerCrossfit[RegressorPipelineDF]:
    # we get the best model_evaluation which is a LGBM - for the sake of test
    # performance
    best_lgbm_evaluation: LearnerEvaluation[RegressorPipelineDF] = [
        evaluation
        for evaluation in regressor_ranker.ranking()
        if isinstance(evaluation.pipeline.regressor, LGBMRegressorDF)
    ][0]

    best_lgbm_regressor: RegressorPipelineDF = best_lgbm_evaluation.pipeline

    return LearnerCrossfit(
        pipeline=best_lgbm_regressor,
        cv=cv_kfold,
        shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=sample)


@pytest.fixture
def feature_names(best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF]) -> Set[str]:
    """
    all unique features across the models in the crossfit, after preprocessing
    """
    return functools.reduce(
        operator.or_, (set(model.features_out) for model in best_lgbm_crossfit.models())
    )


@pytest.fixture
def regressor_inspector(
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF], n_jobs: int
) -> RegressorInspector:
    return RegressorInspector(n_jobs=n_jobs).fit(crossfit=best_lgbm_crossfit)


@pytest.fixture
def simple_preprocessor(sample: Sample) -> TransformerDF:
    return make_simple_transformer(
        impute_median_columns=sample.features.select_dtypes(np.number).columns,
        one_hot_encode_columns=sample.features.select_dtypes(object).columns,
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
def sample(boston_df: pd.DataFrame, boston_target: str, fast_execution: bool) -> Sample:
    return Sample(
        observations=boston_df.iloc[:100, :] if fast_execution else boston_df,
        target=boston_target,
    )


@pytest.fixture
def iris_df(iris_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    iris: Bunch = datasets.load_iris()

    iris_df = pd.DataFrame(
        data=np.c_[iris.data, iris.target], columns=[*iris.feature_names, iris_target]
    )

    # replace target numericals with actual class labels
    iris_df.loc[:, iris_target] = (
        iris_df.loc[:, iris_target].astype(int).map(dict(enumerate(iris.target_names)))
    )

    return iris_df


@pytest.fixture
def iris_sample(iris_df: pd.DataFrame, iris_target: str) -> Sample:
    # the iris dataset
    return Sample(observations=iris_df, target=iris_target)


@pytest.fixture
def iris_sample_binary(iris_sample: Sample) -> Sample:
    # the iris dataset, retaining only two categories so we can do binary classification
    return iris_sample.subsample(
        loc=iris_sample.target.isin(["virginica", "versicolor"])
    )


@pytest.fixture
def iris_sample_binary_dual_target(iris_sample_binary: Sample) -> Sample:
    # the iris dataset, retaining only two categories so we can do binary classification
    target = pd.Series(
        index=iris_sample_binary.index,
        data=pd.Categorical(iris_sample_binary.target).codes,
        name="target",
    )
    return Sample(
        iris_sample_binary.features.join(target).join(target.rename("target2")),
        target=[*iris_sample_binary.target_columns, "target2"],
    )
