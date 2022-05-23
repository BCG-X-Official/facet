import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import datasets
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, KFold
from sklearn.utils import Bunch

from sklearndf import TransformerDF
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression import (
    SVRDF,
    AdaBoostRegressorDF,
    DecisionTreeRegressorDF,
    ExtraTreeRegressorDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
)
from sklearndf.regression.extra import LGBMRegressorDF
from sklearndf.transformation import (
    ColumnTransformerDF,
    OneHotEncoderDF,
    SimpleImputerDF,
)

import facet
from facet.data import Sample
from facet.inspection import LearnerInspector, TreeExplainerFactory
from facet.selection import ModelSelector, MultiEstimatorParameterSpace, ParameterSpace
from facet.validation import BootstrapCV, StratifiedBootstrapCV

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# print the FACET logo
print(facet.__logo__)

# disable SHAP debugging messages
logging.getLogger("shap").setLevel(logging.WARNING)

# configure pandas text output
pd.set_option("display.width", None)  # get display width from terminal
pd.set_option("display.precision", 3)  # 3 digits precision for easier readability

K_FOLDS = 5
N_BOOTSTRAPS = 30

STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


@pytest.fixture
def boston_target() -> str:
    return "price"


@pytest.fixture
def iris_target_name() -> str:
    return "species"


@pytest.fixture
def n_jobs() -> int:
    return -1


@pytest.fixture
def cv_kfold() -> KFold:
    # define a CV
    return KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)


@pytest.fixture
def cv_bootstrap() -> BaseCrossValidator:
    # define a CV
    return BootstrapCV(n_splits=N_BOOTSTRAPS, random_state=42)


@pytest.fixture
def cv_stratified_bootstrap() -> BaseCrossValidator:
    # define a CV
    return StratifiedBootstrapCV(n_splits=N_BOOTSTRAPS, random_state=42)


@pytest.fixture
def regressor_parameters(
    simple_preprocessor: TransformerDF,
) -> MultiEstimatorParameterSpace[RegressorPipelineDF]:
    random_state = {"random_state": 42}

    space_1 = ParameterSpace(
        RegressorPipelineDF(
            preprocessing=simple_preprocessor, regressor=LGBMRegressorDF(**random_state)
        )
    )
    space_1.regressor.max_depth = [5, 10]
    space_1.regressor.min_split_gain = [0.1, 0.2]
    space_1.regressor.num_leaves = [50, 100, 200]

    space_2 = ParameterSpace(
        RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=AdaBoostRegressorDF(**random_state),
        )
    )
    space_2.regressor.n_estimators = [50, 80]

    space_3 = ParameterSpace(
        RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=RandomForestRegressorDF(**random_state),
        )
    )
    space_3.regressor.n_estimators = [50, 80]

    space_4 = ParameterSpace(
        RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=DecisionTreeRegressorDF(**random_state),
        )
    )
    space_4.regressor.max_depth = [0.5, 1.0]
    space_4.regressor.max_features = [0.5, 1.0]

    space_5 = ParameterSpace(
        RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=ExtraTreeRegressorDF(**random_state),
        )
    )
    space_5.regressor.max_depth = [5, 10, 12]

    space_6 = ParameterSpace(
        RegressorPipelineDF(preprocessing=simple_preprocessor, regressor=SVRDF())
    )
    space_6.regressor.gamma = [0.5, 1]
    space_6.regressor.C = [50, 100]

    space_7 = ParameterSpace(
        RegressorPipelineDF(
            preprocessing=simple_preprocessor, regressor=LinearRegressionDF()
        )
    )
    space_7.regressor.normalize = [False, True]

    return MultiEstimatorParameterSpace(
        space_1,
        space_2,
        space_3,
        space_4,
        space_5,
        space_6,
        space_7,
    )


@pytest.fixture
def regressor_selector(
    cv_kfold: KFold,
    regressor_parameters: MultiEstimatorParameterSpace[RegressorPipelineDF],
    sample: Sample,
    n_jobs: int,
) -> ModelSelector[RegressorPipelineDF, GridSearchCV]:
    return ModelSelector(
        searcher_type=GridSearchCV,
        parameter_space=regressor_parameters,
        cv=cv_kfold,
        scoring="r2",
        n_jobs=n_jobs,
    ).fit(sample=sample)


PARAM_CANDIDATE__ = "param_candidate__"


@pytest.fixture
def best_lgbm_model(
    regressor_selector,
    sample: Sample,
) -> RegressorPipelineDF:
    # we get the best model_evaluation which is a LGBM - for the sake of test
    # performance
    # noinspection PyTypeChecker
    best_lgbm_params: Dict[str, Any] = (
        pd.DataFrame(regressor_selector.searcher_.cv_results_)
        .pipe(
            lambda df: df.loc[df.loc[:, "param_candidate_name"] == "LGBMRegressorDF", :]
        )
        .pipe(lambda df: df.loc[df.loc[:, "rank_test_score"].idxmin(), "params"])
    )

    len_param_candidate = len(PARAM_CANDIDATE__)
    return (
        best_lgbm_params["candidate"]
        .clone()
        .set_params(
            **{
                param[len_param_candidate:]: value
                for param, value in best_lgbm_params.items()
                if param.startswith(PARAM_CANDIDATE__)
            }
        )
        .fit(X=sample.features, y=sample.target)
    )


@pytest.fixture
def preprocessed_feature_names(best_lgbm_model: RegressorPipelineDF) -> Set[str]:
    """
    Names of all features after preprocessing
    """
    return set(best_lgbm_model.feature_names_out_)


@pytest.fixture
def regressor_inspector(
    best_lgbm_model: RegressorPipelineDF, sample: Sample, n_jobs: int
) -> LearnerInspector:
    inspector = LearnerInspector(
        pipeline=best_lgbm_model,
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", uses_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(sample=sample)

    return inspector


@pytest.fixture
def simple_preprocessor(sample: Sample) -> TransformerDF:
    features = sample.features

    column_transforms: List[Tuple[str, Any, Any]] = []

    numeric_columns: pd.Index = features.select_dtypes(np.number).columns
    if numeric_columns is not None and len(numeric_columns) > 0:
        column_transforms.append(
            (
                STEP_IMPUTE,
                SimpleImputerDF(strategy="median"),
                list(map(str, numeric_columns)),
            )
        )

    category_columns = features.select_dtypes(object).columns
    if category_columns is not None and len(category_columns) > 0:
        column_transforms.append(
            (
                STEP_ONE_HOT_ENCODE,
                OneHotEncoderDF(sparse=False, handle_unknown="ignore"),
                list(map(str, category_columns)),
            )
        )

    return ColumnTransformerDF(transformers=column_transforms)


@pytest.fixture
def boston_df(boston_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

    return pd.DataFrame(
        data=np.c_[boston.data, boston.target],
        columns=[*boston.feature_names, boston_target],
    )


@pytest.fixture
def sample(boston_df: pd.DataFrame, boston_target: str) -> Sample:
    return Sample(observations=boston_df.iloc[:100, :], target_name=boston_target)


@pytest.fixture
def iris_df(iris_target_name: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    iris: Bunch = datasets.load_iris()

    iris_df = pd.DataFrame(
        data=np.c_[iris.data, iris.target],
        columns=[*iris.feature_names, iris_target_name],
    )

    # replace target numericals with actual class labels
    iris_df.loc[:, iris_target_name] = (
        iris_df.loc[:, iris_target_name]
        .astype(int)
        .map(dict(enumerate(iris.target_names)))
    )

    return iris_df


@pytest.fixture
def iris_sample_multi_class(iris_df: pd.DataFrame, iris_target_name: str) -> Sample:
    # the iris dataset
    return Sample(
        observations=iris_df.assign(weight=2.0),
        target_name=iris_target_name,
        weight_name="weight",
    )


@pytest.fixture
def iris_sample_binary(iris_sample_multi_class) -> Sample:
    # the iris dataset, retaining only two categories so we can do binary classification
    return iris_sample_multi_class.subsample(
        loc=iris_sample_multi_class.target.isin(["virginica", "versicolor"])
    )


@pytest.fixture
def iris_sample_binary_dual_target(
    iris_sample_binary: Sample, iris_target_name
) -> Sample:
    # the iris dataset, retaining only two categories so we can do binary classification
    target = pd.Series(
        index=iris_sample_binary.index,
        data=pd.Categorical(iris_sample_binary.target).codes,
        name=iris_target_name,
    )
    iris_target_2 = f"{iris_target_name}2"
    return Sample(
        iris_sample_binary.features.join(target).join(target.rename(iris_target_2)),
        target_name=[iris_sample_binary.target_name, iris_target_2],
    )


COL_PARAM = "param"
COL_CANDIDATE = "candidate"
COL_CANDIDATE_NAME = "candidate_name"
COL_CLASSIFIER = "classifier"
COL_REGRESSOR = "regressor"
COL_SCORE = ("score", "test", "mean")


def check_ranking(
    ranking: pd.DataFrame,
    is_classifier: bool,
    scores_expected: Sequence[float],
    params_expected: Optional[Mapping[int, Mapping[str, Any]]],
    candidate_names_expected: Optional[Sequence[str]] = None,
) -> None:
    """
    Test helper to check rankings produced by learner rankers.

    :param ranking: summary data frame
    :param is_classifier: flag if ranking was performed on classifiers, or regressors
    :param scores_expected: expected ranking scores, rounded to 3 decimal places
    :param params_expected: expected learner parameters
    :param candidate_names_expected: optional list of expected learners;
        only required for multi estimator search
    """

    scores_actual: pd.Series = ranking.loc[:, COL_SCORE].values[: len(scores_expected)]

    assert_allclose(
        scores_actual,
        scores_expected,
        rtol=0.01,
        err_msg=(
            f"unexpected scores: got {scores_actual} but expected {scores_expected}"
        ),
    )

    col_learner = COL_CLASSIFIER if is_classifier else COL_REGRESSOR

    if params_expected is not None:
        param_columns: pd.DataFrame = ranking.loc[:, (COL_PARAM, col_learner)]
        for rank, parameters_expected in params_expected.items():
            parameters_actual: Dict[str, Any] = (
                param_columns.iloc[rank, :].dropna().to_dict()
            )
            assert parameters_actual == parameters_expected, (
                f"unexpected parameters for learner at rank #{rank}: "
                f"got {parameters_actual} but expected {parameters_expected}"
            )

    if candidate_names_expected:
        candidates_actual: np.ndarray = ranking.loc[
            :, (COL_CANDIDATE_NAME, "-", "-")
        ].values[: len(candidate_names_expected)]
        assert_array_equal(
            candidates_actual,
            candidate_names_expected,
            (
                f"unexpected candidate names: got {list(candidates_actual)} "
                f"but expected {list(candidate_names_expected)}"
            ),
        )


@pytest.fixture
def iris_classifier_selector_binary(
    iris_sample_binary: Sample,
    cv_stratified_bootstrap: StratifiedBootstrapCV,
    n_jobs: int,
) -> ModelSelector[ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV]:
    return fit_classifier_selector(
        sample=iris_sample_binary, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_selector_multi_class(
    iris_sample_multi_class: Sample,
    cv_stratified_bootstrap: StratifiedBootstrapCV,
    n_jobs: int,
) -> ModelSelector[ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV]:
    return fit_classifier_selector(
        sample=iris_sample_multi_class, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_selector_dual_target(
    iris_sample_binary_dual_target: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> ModelSelector[ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV]:
    return fit_classifier_selector(
        sample=iris_sample_binary_dual_target, cv=cv_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_binary(
    iris_classifier_selector_binary: ModelSelector[ClassifierPipelineDF, GridSearchCV],
) -> ClassifierPipelineDF[RandomForestClassifierDF]:
    return iris_classifier_selector_binary.best_estimator_


@pytest.fixture
def iris_classifier_multi_class(
    iris_classifier_selector_multi_class: ModelSelector[
        ClassifierPipelineDF, GridSearchCV
    ],
) -> ClassifierPipelineDF[RandomForestClassifierDF]:
    return iris_classifier_selector_multi_class.best_estimator_


@pytest.fixture
def iris_inspector_multi_class(
    iris_classifier_multi_class: ClassifierPipelineDF[RandomForestClassifierDF],
    iris_sample_multi_class: Sample,
    n_jobs: int,
) -> LearnerInspector[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return LearnerInspector(
        pipeline=iris_classifier_multi_class, shap_interaction=True, n_jobs=n_jobs
    ).fit(sample=iris_sample_multi_class)


#
# Utility functions
#


def fit_classifier_selector(
    sample: Sample, cv: BaseCrossValidator, n_jobs: int
) -> ModelSelector[ClassifierPipelineDF[RandomForestClassifierDF], GridSearchCV]:
    # define the parameter space
    parameter_space = ParameterSpace(
        ClassifierPipelineDF(
            classifier=RandomForestClassifierDF(random_state=42),
            preprocessing=None,
        )
    )
    parameter_space.classifier.n_estimators = [10, 50]
    parameter_space.classifier.min_samples_leaf = [4, 8]

    # pipeline inspector only supports binary classification,
    # therefore filter the sample down to only 2 target classes
    return ModelSelector(
        searcher_type=GridSearchCV,
        parameter_space=parameter_space,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
    ).fit(sample=sample)
