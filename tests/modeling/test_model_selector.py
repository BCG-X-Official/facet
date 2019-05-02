from tests.shared_fixtures import test_sample as test_sample_data
from yieldengine.loading.sample import Sample
from yieldengine.preprocessing.cross_validation import CircularCrossValidator
from yieldengine.modeling.model_selector import ModelSelector
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm.sklearn import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


def test_model_selector(test_sample_data):
    # drop columns that should not take part in modeling
    test_sample_data = test_sample_data.drop(columns=["Date", "Batch Id"])

    # replace values of +/- infinite with n/a, then drop all n/a columns:
    test_sample_data = test_sample_data.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="all"
    )

    sample = Sample(sample=test_sample_data, target="Yield")

    # define the circular cross validator with just 5 folds (to speed up testing)
    circular_cv = CircularCrossValidator(
        num_samples=len(sample), test_ratio=0.20, num_folds=5
    )

    # define a ColumnTransformer to pre-process:
    preprocessor = ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="mean"), sample.numerical_features),
            (
                "categorical",
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                sample.categorical_features,
            ),
        ]
    )

    # define a sklearn Pipeline, containing the preprocessor defined above:
    pre_pipeline = Pipeline([("prep", preprocessor)])

    # run fit_transform once to assure it works:
    pre_pipeline.fit_transform(sample.feature_data)

    # define all grid-searchers, keep in this list:
    searchers = []

    models = [
        LGBMRegressor(),
        AdaBoostRegressor(),
        RandomForestRegressor(),
        DecisionTreeRegressor(),
        ExtraTreeRegressor(),
        SVR(),
        LinearRegression(),
    ]

    params = [
        # lgbm
        {
            "max_depth": (5, 10),
            "min_split_gain": (0.1, 0.2),
            "num_leaves": (50, 100, 200),
        },
        # adaboost
        {"n_estimators": (50, 80)},
        # random forest
        {"n_estimators": (50, 80)},
        # decisiontree
        {"max_depth": (0.5, 1.0), "max_features": (0.5, 1.0)},
        # extratree
        {"max_depth": (5, 10, 12)},
        # svr
        {"gamma": (0.5, 1), "C": (50, 100)},
        # linear reg
        {"normalize": (False, True)},
    ]

    for model, params in zip(models, params):
        search = GridSearchCV(
            estimator=model,
            cv=circular_cv.get_train_test_splits_as_indices(),
            param_grid=params,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            n_jobs=4,
        )
        searchers.append(search)

    # instantiate the model selector
    ms = ModelSelector(searchers=searchers, preprocessing=pre_pipeline)

    # run train_models
    ms.train_models(sample=sample)

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
