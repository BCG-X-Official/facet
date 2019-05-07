from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
import pandas as pd
from typing import *

from sklearn.base import BaseEstimator

from yieldengine.loading.sample import Sample


class Model(NamedTuple):
    name: str
    estimator: BaseEstimator
    parameters: Dict[str, Any]


class Searcher(NamedTuple):
    model: Model
    grid_search: GridSearchCV


class ModelZoo:
    __slots__ = ["__models"]

    def __init__(self) -> None:
        self.__models = list()

    def add_model(
        self, name: str, estimator: BaseEstimator, parameters: Dict[str, Any]
    ) -> "ModelZoo":
        m = Model(name=name, estimator=estimator, parameters=parameters)
        self.__models.append(m)
        return self

    @property
    def models(self) -> List[Model]:
        return self.__models


class ModelPipeline:
    def __init__(
        self, zoo: ModelZoo, preprocessing: Pipeline = None, cv=None, scoring=None
    ) -> None:
        """
        Constructs a ModelSelector

        :param searchers: a list of scikit-learn GridSearchCV objects (not fitted)
        :param preprocessing: a scikit-learn Pipeline that should be used as a preprocessor
        :return None
        """
        self.__model_zoo = zoo
        self.__preprocessing = preprocessing
        self.__pipeline = None
        self.__cv = cv
        self.__scoring = scoring

        self.__searchers = searchers = self.__construct_searchers(zoo, cv, scoring)
        self.__pipeline = self.__construct_pipeline(preprocessing, searchers)

    @staticmethod
    def __construct_searchers(zoo: ModelZoo, cv, scoring) -> List[Searcher]:
        searchers = list()

        for model in zoo.models:
            search = GridSearchCV(
                estimator=model.estimator,
                cv=cv,
                param_grid=model.parameters,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )
            searchers.append(Searcher(model, search))

        return searchers

    @staticmethod
    def __construct_pipeline(preprocessing, searchers) -> Pipeline:
        """
        Constructs and returns a single scikit-learn Pipeline, comprising of all given searchers and using the
        (if supplied) preprocessing step.

        All given :code:`searchers` will be included into the pipeline as a parallel step leveraging
        :code:`FeatureUnion`. This means, calling :code:`fit(X, y)` on it will run all gridsearchers, which you then
        can inspect using :code:`model_selector.rank_models()`, :code:`model_selector.rank_model_instances()` etc.

        Running :code:`transform(X)` on the pipeline will yield a (#samples X #searchers) shaped numpy array
        with predictions that each model made.

        :return: Pipeline
        """

        class ModelTransformer(TransformerMixin):
            def __init__(self, model):
                self.model = model

            def fit(self, *args, **kwargs):
                self.model.fit(*args, **kwargs)
                return self

            def transform(self, X, **transform_params):
                return pd.DataFrame(self.model.predict(X))

        # generate a list of (name, obj) tuples for all given estimators (=GridSearchCV objs.):
        #   name: simply e0, e1, e2, ...
        #   obj: the GridSearchCV wrapped in a ModelTransformer() object, to be feature union compliant
        estimator_steps = [
            (f"e_{model.name}", ModelTransformer(grid_search))
            for model, grid_search in searchers
        ]

        # with the above created list of ModelTransformers, create FeatureUnion()
        # benefit: can be evaluated in parallel and pre-processing is shared
        est_feature_union = (
            "estimators",
            FeatureUnion(transformer_list=estimator_steps),
        )

        # if pre-processing pipeline was given, insert it into the pipeline:
        if preprocessing is not None:
            return Pipeline([("preprocessing", preprocessing), est_feature_union])
        else:
            # if pre-processing pipeline was not given, create a minimal Pipeline with just the estimators
            return Pipeline([est_feature_union])

    def run(self, sample: Sample) -> None:
        """
        Execute the pipeline with the given sample
        :param sample: sample to fit pipeline to
        """
        self.pipeline.fit(X=sample.features, y=sample.target)

    @property
    def pipeline(self):
        return self.__pipeline

    @property
    def searchers(self):
        return self.__searchers

    @property
    def model_zoo(self):
        return self.__model_zoo


class ModelRanker:
    """
    Class that helps in training, validating and ranking multiple scikit-learn models
    (i.e. various regressor implementations), that depend all on the same (or none) pre-processing pipeline

    For an example, please see:
    `Example: Loading, Preprocessing, Circular CV, Model Selection  <./examples/e1.html>`_
    """

    def __init__(self, model_pipeline: ModelPipeline) -> None:
        """
        Constructs a ModelSelector

        :param model_pipeline:
        :return None
        """
        self.__model_pipeline = model_pipeline

    def rank_models(self) -> pd.DataFrame:
        all_cv_results = None

        for model, search in self.__model_pipeline.searchers:
            if all_cv_results is None:
                all_cv_results = pd.DataFrame(search.cv_results_)
                all_cv_results["estimator"] = model.estimator
                all_cv_results["model_name"] = model.name
            else:
                new_cv_results = pd.DataFrame(search.cv_results_)
                new_cv_results["estimator"] = model.estimator
                new_cv_results["model_name"] = model.name

                all_cv_results = all_cv_results.append(new_cv_results, sort=False)

        all_cv_results["final_score"] = (
            all_cv_results["mean_test_score"] - 2 * all_cv_results["std_test_score"]
        )

        all_cv_results = all_cv_results[
            [
                "model_name",
                "estimator",
                "params",
                "mean_test_score",
                "std_test_score",
                "final_score",
                "mean_fit_time",
            ]
        ]

        all_cv_results = all_cv_results.sort_values(
            by="final_score", ascending=False
        ).reset_index(drop=True)

        return all_cv_results

    def summary_string(self, limit: int = 25) -> str:
        """
        Generates a summary string of the best models, ranked by :code:`rank_modls()`

        :param limit: How many ranks to max. output

        :return: str
        """
        return "\n".join(
            [
                f" Rank {i + 1}: {m.model_name}, "
                f"Score: {-1 * m.final_score}, Params: {m.params}"
                for i, m in self.rank_models().iterrows()
                if i < limit
            ]
        )
