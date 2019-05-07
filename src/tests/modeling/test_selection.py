import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

# noinspection PyUnresolvedReferences
from tests.shared_fixtures import batch_table
from yieldengine.loading.sample import Sample
from yieldengine.modeling.selection import ModelPipeline, ModelRanker, ModelZoo, Model
from yieldengine.modeling.validation import CircularCrossValidator


def test_model_selector(batch_table: pd.DataFrame) -> None:

    # drop columns that should not take part in modeling
    batch_table = batch_table.drop(columns=["Date", "Batch Id"])

    # replace values of +/- infinite with n/a, then drop all n/a columns:
    batch_table = batch_table.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="all"
    )

    sample = Sample(observations=batch_table, target_name="Yield")

    # define the circular cross validator with just 5 folds (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_folds=5)

    # define a ColumnTransformer to pre-process:
    preprocessor = ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="mean"), sample.features_numerical),
            (
                "categorical",
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                sample.features_categorical,
            ),
        ]
    )

    # define a sklearn Pipeline, containing the preprocessor defined above:
    pre_pipeline = Pipeline([("prep", preprocessor)])

    # run fit_transform once to assure it works:
    pre_pipeline.fit_transform(sample.features)

    model_zoo = ModelZoo(
        [
            Model(
                name="lgbm",
                estimator=LGBMRegressor(),
                parameters={
                    "max_depth": (5, 10),
                    "min_split_gain": (0.1, 0.2),
                    "num_leaves": (50, 100, 200),
                },
            ),
            Model(
                name="ada",
                estimator=AdaBoostRegressor(),
                parameters={"n_estimators": (50, 80)},
            ),
            Model(
                name="rf",
                estimator=RandomForestRegressor(),
                parameters={"n_estimators": (50, 80)},
            ),
            Model(
                name="dt",
                estimator=DecisionTreeRegressor(),
                parameters={"max_depth": (0.5, 1.0), "max_features": (0.5, 1.0)},
            ),
            Model(
                name="et",
                estimator=ExtraTreeRegressor(),
                parameters={"max_depth": (5, 10, 12)},
            ),
            Model(
                name="svr",
                estimator=SVR(),
                parameters={"gamma": (0.5, 1), "C": (50, 100)},
            ),
            Model(
                name="lr",
                estimator=LinearRegression(),
                parameters={"normalize": (False, True)},
            ),
        ]
    )

    mp: ModelPipeline = ModelPipeline(
        zoo=model_zoo,
        preprocessing=pre_pipeline,
        cv=circular_cv,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
    )

    # train the models
    mp.pipeline.fit(sample.features, sample.target)

    # instantiate the model selector
    ms: ModelRanker = ModelRanker(model_pipeline=mp)

    # when done, get ranking
    ranked_models = ms.rank_models()
    # check types
    assert type(ranked_models) == list
    assert type(ranked_models[0]) == GridSearchCV
    # check sorting
    assert (
        ranked_models[0].best_score_
        >= ranked_models[1].best_score_
        >= ranked_models[2].best_score_
    )

    # print summary:
    print("Ranked models:")
    print(ms.summary_string())

    # get ranked model-instances:
    ranked_model_instances = ms.rank_model_instances(n_best_ranked=3)

    # check data structure
    assert type(ranked_model_instances) == list
    assert type(ranked_model_instances[0]) == dict
    assert {"score", "estimator", "params"} == ranked_model_instances[0].keys()

    # check sorting
    assert (
        ranked_model_instances[0]["score"]
        >= ranked_model_instances[1]["score"]
        >= ranked_model_instances[2]["score"]
    )

    # test transform():
    assert mp.pipeline.transform(sample.features).shape == (
        len(sample),
        len(list(mp.searchers())),
    )


def test_model_selector_no_preprocessing() -> None:
    # filter out warnings triggerd by sk-learn/numpy
    import warnings

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")
    from sklearn import datasets, svm

    # load example data
    iris = datasets.load_iris()

    # define a yield-engine circular CV:
    my_cv = CircularCrossValidator(test_ratio=0.21, num_folds=50)

    # define parameters and model
    models = ModelZoo(
        [
            Model(
                name="svc",
                estimator=svm.SVC(gamma="scale"),
                parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
            )
        ]
    )

    mp: ModelPipeline = ModelPipeline(zoo=models, cv=my_cv)

    # train the models
    mp.pipeline.fit(iris.data, iris.target)

    # instantiate the model selector
    ms: ModelRanker = ModelRanker(model_pipeline=mp)

    print(pd.DataFrame(ms.rank_model_instances()).head())
