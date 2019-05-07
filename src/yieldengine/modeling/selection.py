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

    __slots__ = ["__models"]

    def __init__(self) -> None:
        """
        Initializes an empty model zoo.
        """
        self.__models = list()

    def add_model(
        self, estimator: BaseEstimator, parameter_grid: Dict[str, Any]
    ) -> "ModelZoo":
        """
        Add another model into this model zoo. Supports chaining.

        :param estimator: an estimator to add into the zoo
        :param parameter_grid: a parameter grid for the given estimator

        :return: the expanded model zoo
        """
        m = Model(estimator=estimator, parameter_grid=parameter_grid)
        self.__models.append(m)
        return self

    @property
    def models(self) -> List[Model]:
        """
        Property of ModelZoo

        :return: a list of :code:`yieldengine.modeling.selection.Model` instances
        """
        return self.__models


class Scoring:
    """
    Class for scoring related logic.
    """

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

        for model in zoo.models:
            search = GridSearchCV(
                estimator=model.estimator,
                cv=cv,
                param_grid=model.parameter_grid,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
            )
            searchers.append(search)

        return searchers

    @staticmethod
    def __construct_pipeline(preprocessing, searchers) -> Pipeline:

        # helper class to view a model (Estimator/GridSearchCV) as a Transformer:
        class ModelTransformer(TransformerMixin):
            def __init__(self, model):
                self.model = model

            def fit(self, *args, **kwargs):
                self.model.fit(*args, **kwargs)
                return self

            def transform(self, X, **transform_params):
                return pd.DataFrame(self.model.predict(X))

        # generate a list of (name, obj) tuples for all estimators (=GridSearchCV objs.)
        #   name: simply e0, e1, e2, ...
        #   obj: the GridSearchCV wrapped in a ModelTransformer() object, to be
        #        feature union compliant
        estimator_steps = [
            (f"e{i}", ModelTransformer(grid_search))
            for i, grid_search in enumerate(searchers)
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

    def run(
        self, sample: Sample, ranking_scorer: Callable = Scoring.default_ranking_scorer
    ) -> "ModelRanking":
        """
        Execute the pipeline with the given sample and return the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models

        :return the created model ranking of type :code:`ModelRanking`

        """
        self.pipeline.fit(X=sample.features, y=sample.target)
        return ModelRanking(ranker=self, ranking_scorer=ranking_scorer)

    @property
    def pipeline(self):
        """
        Property of ModelRanker

        :return: the complete scikit-learn pipeline
        """
        return self.__pipeline

    @property
    def searchers(self):
        """
        Property of ModelRanker

        :return: the constructed GridSearchCV instances
        """
        return self.__searchers


class ModelRanking:
    """
    Turns the output of a ModelRanker into a ranked list of model instances, which we
    denote as the combination of (estimator, parameters). Each of these are captured
    using the `RankedModel` class.

    """

    def __init__(
        self,
        ranker: ModelRanker,
        ranking_scorer: Callable = Scoring.default_ranking_scorer,
    ):
        """
        Turn the output of a ModelRanker into a ModelRanking

        :param ranker: a ModelRanker that was executed
        :param ranking_scorer: (optional) a custom scoring function to score across the
        model zoo. The default is :code:`Scoring.default_ranking_scorer`
        """
        self.__ranking = self.__construct_ranking(ranker, ranking_scorer)

    @staticmethod
    def __construct_ranking(
        ranker: ModelRanker, ranking_scorer: Callable
    ) -> List[RankedModel]:

        # consolidate results of all searchers into "results"
        results = list()

        for search in ranker.searchers:
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
                    search.cv_results_["params"],
                    search.cv_results_["mean_test_score"],
                    search.cv_results_["std_test_score"],
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

        return ranking

    def get_rank(self, rank: int = BEST_MODEL_RANK) -> RankedModel:
        """
        Returns the model instance at a given rank.

        :param rank: the rank of the model to get
        
        :return: a RankedModel instance
        """
        return self.__ranking[rank]

    def summary_string(self, limit: int = 25) -> str:
        """
        Generates a summary string of the best model instances

        :param limit: How many ranks to max. output

        :return: str
        """
        return "\n".join(
            [
                f" Rank {mr.rank + 1}: {mr.estimator.__class__}, "
                f"Score: {mr.score}, Params: {mr.parameters}"
                for mr in self.__ranking
                if mr.rank < limit
            ]
        )

    def __len__(self) -> int:
        return len(self.__ranking)

    def __str__(self):
        return self.summary_string()
