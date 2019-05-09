from copy import deepcopy
from typing import *

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from yieldengine.loading.sample import Sample

BEST_MODEL_RANK = 0


class Model(NamedTuple):
    estimator: BaseEstimator
    parameter_grid: Dict[str, Any]


class RankedModel(NamedTuple):
    estimator: BaseEstimator
    parameters: Dict[str, Any]
    score: float
    rank: int


class ModelZoo:
    """
    Class to register models (of type: BaseEstimator) and parameter grids.
    """

    __slots__ = ["_models"]

    def __init__(self, models: Iterable[Model]) -> None:
        self._models = list(models)

    def models(self) -> Iterable[Model]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)


class ModelRanker:
    """
    Turns a model zoo along with

        - a (optional) pre-processing pipeline
        - a cross-validation instance
        - a scoring function
    into a scikit-learn pipeline.

    :param zoo: a model zoo
    :param preprocessing: a scikit-learn Pipeline that should be used as a \
    preprocessor (optional)
    :param cv: a cross validation object (i.e. CircularCrossValidator)
    :param scoring: a scorer to use when doing CV within GridSearch

    """

    F_PARAMETERS = "params"
    F_MEAN_TEST_SCORE = "mean_test_score"
    F_SD_TEST_SCORE = "std_test_score"

    def __init__(
        self, zoo: ModelZoo, preprocessing: Pipeline = None, cv=None, scoring=None
    ) -> None:
        self.__model_zoo = zoo
        self.__preprocessing = preprocessing
        self.__pipeline = None
        self.__cv = cv
        self.__scoring = scoring

        self.__searchers = searchers = self.__construct_searchers(zoo, cv, scoring)
        self.__pipeline = self.__construct_pipeline(preprocessing, searchers)

    @staticmethod
    def __construct_searchers(zoo: ModelZoo, cv, scoring) -> List[GridSearchCV]:
        searchers = list()

        for model in zoo.models():
            search = GridSearchCV(
                estimator=model.estimator,
                cv=cv,
                param_grid=model.parameter_grid,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
                refit=False,
            )
            searchers.append(search)

        return searchers

    @staticmethod
    def __construct_pipeline(
        preprocessing: Pipeline, estimators: List[BaseEstimator]
    ) -> Pipeline:

        # helper class to view a model (Estimator/GridSearchCV) as a Transformer:
        class ModelTransformer(TransformerMixin):
            def __init__(self, model) -> None:
                self.model = model

            def fit(self, *args, **kwargs) -> "ModelTransformer":
                self.model.fit(*args, **kwargs)
                return self

            def transform(self, X, **transform_params) -> pd.DataFrame:
                return pd.DataFrame(self.model.predict(X))

        # generate a list of (name, obj) tuples for all estimators (or GridSearchCV
        # objs.)
        #   name: simply e0, e1, e2, ...
        #   obj: the Estimator/GridSearchCV wrapped in a ModelTransformer() object,
        #   to be feature union compliant
        estimator_steps = [
            (f"e{i}", ModelTransformer(estimator))
            for i, estimator in enumerate(estimators)
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
            # if pre-processing pipeline was not given, create a minimal Pipeline
            # with just the estimators
            return Pipeline([est_feature_union])

    @staticmethod
    def default_ranking_scorer(mean_test_score, std_test_score) -> float:
        """
        The default scoring function to evaluate on top of GridSearchCV test scores,
        given by :code:`GridSearchCV.cv_results_`.

        Its output is used for ranking globally across the model zoo.

        :param mean_test_score: the mean test score across all folds for a (estimator,\
        parameters) combination
        :param std_test_score: the standard deviation of test scores across all folds \
        for a (estimator, parameters) combination

        :return: final score for a (estimator, parameters) combination
        """
        return mean_test_score - 2 * std_test_score

    def run(
        self, sample: Sample, ranking_scorer: Callable = None, refit=True
    ) -> "ModelRanking":
        """
        Execute the pipeline with the given sample and return the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models
        :param refit: whether to refit estimators on whole dataset

        :return the created model ranking of type :code:`ModelRanking`

        """
        self.pipeline.fit(X=sample.features, y=sample.target)

        model_ranking = self.rank_models(
            searchers=self.__searchers, ranking_scorer=ranking_scorer
        )

        if refit:
            # we build a new pipeline that fits all estimators (that have their params
            # already set), without CV and without GridSearching:
            refit_pipeline = self.__construct_pipeline(
                preprocessing=self.__preprocessing,
                estimators=[m.estimator for m in model_ranking],
            )
            # call fit on the new pipeline:
            refit_pipeline.fit(X=sample.features, y=sample.target)

        return model_ranking

    @staticmethod
    def rank_models(
        searchers: List[GridSearchCV], ranking_scorer: Callable
    ) -> "ModelRanking":
        """

        :param searchers: GridSearchCV instances across which to rank
        :param ranking_scorer: (optional) a custom scoring function to score across the
        model zoo. The default is :code:`ModelRanker.default_ranking_scorer`
        :return: an instance of ModelRanking that wraps the results
        """

        if ranking_scorer is None:
            ranking_scorer = ModelRanker.default_ranking_scorer

        # consolidate results of all searchers into "results"
        results = list()

        for search in searchers:
            search_results = [
                (
                    # note: we have to copy the estimator, to ensure it will actually
                    # retain the parameters we set for each row in separate objects..
                    deepcopy(search.estimator.set_params(**params)),
                    params,
                    # compute the final score using function defined above:
                    ranking_scorer(mean_test_score, std_test_score),
                )
                # we read and iterate over these 3 attributes from cv_results_:
                for (params, mean_test_score, std_test_score) in zip(
                    search.cv_results_[ModelRanker.F_PARAMETERS],
                    search.cv_results_[ModelRanker.F_MEAN_TEST_SCORE],
                    search.cv_results_[ModelRanker.F_SD_TEST_SCORE],
                )
            ]

            results.extend(search_results)

        # sort the results list by value at index 2 -> computed final score
        results.sort(key=lambda r: r[2] * -1)

        # create ranking by assigning rank values and creating "RankedModel" types
        ranking = [
            RankedModel(estimator=r[0], parameters=r[1], score=r[2], rank=i)
            for i, r in enumerate(results)
        ]

        return ModelRanking(ranking=ranking)

    @property
    def pipeline(self) -> Pipeline:
        """
        Property of ModelRanker

        :return: the complete scikit-learn pipeline
        """
        return self.__pipeline

    def searchers(self) -> Iterable[GridSearchCV]:
        return iter(self.__searchers)

    @property
    def model_zoo(self) -> ModelZoo:
        return self.__model_zoo


class ModelRanking:
    """
    Utility class that wraps a list of RankedModel

    """

    def __init__(self, ranking: List[RankedModel]):
        """
        Utility class that wraps a list of RankedModel

        :param ranking: the list of RankedModel instances this ranking is based on
        """
        self.__ranking = ranking

    def model(self, rank: int = BEST_MODEL_RANK) -> RankedModel:
        """
        Returns the model instance at a given rank.

        :param rank: the rank of the model to get
        
        :return: a RankedModel instance
        """
        return self.__ranking[rank]

    def summary_report(self, limit: int = 10) -> str:
        """
        Generates a summary string of the best model instances

        :param limit: How many ranks to max. output

        :return: str
        """

        rows = [
            (
                ranked_model.rank,
                ranked_model.estimator.__class__.__name__,
                ranked_model.score,
                ranked_model.parameters,
            )
            for ranked_model in self.__ranking[:limit]
        ]

        name_width = max([len(row[1]) for row in rows])

        return "\n".join(
            [
                f" Rank {row[0]:2d}:'"
                f"{row[1]:{name_width}s}, "
                f"Score: {row[2]:.2e}, "
                f"Params: {row[3]}"
                for row in rows
            ]
        )

    def __len__(self) -> int:
        return len(self.__ranking)

    def __str__(self) -> str:
        return self.summary_report()

    def __iter__(self) -> Iterable[RankedModel]:
        return iter(self.__ranking)
