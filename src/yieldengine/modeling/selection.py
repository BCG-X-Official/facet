from typing import *

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.pipeline import Pipeline

from yieldengine.loading.sample import Sample
from yieldengine.modeling.factory import ModelPipelineFactory


class Model(NamedTuple):
    estimator: BaseEstimator
    parameter_grid: Dict[str, Any]


class RankedModel(NamedTuple):
    estimator: BaseEstimator
    parameters: Dict[str, Any]
    test_score_mean: float
    test_score_std: float
    ranking_score: float


class ModelRanker:
    """
    Turns a model zoo along with

        - a (optional) pre-processing pipeline
        - a cross-validation instance
        - a scoring function
    into a scikit-learn pipeline.

    :param models: list of model grids to be ranked
    :param pipeline_factory: a pipeline factory, e.g., to insert a preprocessor
    :param cv: a cross validation object (i.e. CircularCrossValidator)
    :param scoring: a scorer to use when doing CV within GridSearch
    """

    __slots__ = ["_models", "_scoring", "_cv", "_searchers", "_pipeline"]

    F_PARAMETERS = "params"
    F_MEAN_TEST_SCORE = "mean_test_score"
    F_SD_TEST_SCORE = "std_test_score"

    def __init__(
        self,
        models: Iterable[Model],
        pipeline_factory: ModelPipelineFactory = None,
        cv: BaseCrossValidator = None,
        scoring=None,
    ) -> None:
        self._models = list(models)
        self._cv = cv
        self._scoring = scoring

        # construct searchers
        self._searchers = searchers = [
            GridSearchCV(
                estimator=model.estimator,
                cv=cv,
                param_grid=model.parameter_grid,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1,
                refit=False,
            )
            for model in self._models
        ]

        self._pipeline = (
            ModelPipelineFactory() if pipeline_factory is None else pipeline_factory
        ).make_pipeline(estimators=searchers)

    @staticmethod
    def default_ranking_scorer(mean_test_score: float, std_test_score: float) -> float:
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
        self, sample: Sample, ranking_scorer: Callable[[float, float], float] = None
    ) -> "ModelRanking":
        """
        Execute the pipeline with the given sample and return the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models,
        taking mean and standard deviation of the ranking scores and returning the
        overall ranking score (default: ModelRanking.default_ranking_scorer)

        :return the created model ranking of type :code:`ModelRanking`

        """
        self.pipeline.fit(X=sample.features, y=sample.target)

        if ranking_scorer is None:
            ranking_scorer = ModelRanker.default_ranking_scorer

        # consolidate results of all searchers into "results"
        search_results = [
            RankedModel(
                # note: we have to copy the estimator, to ensure it will actually
                # retain the parameters we set for each row in separate objects..
                estimator=clone(search.estimator).set_params(**params),
                parameters=params,
                test_score_mean=test_score_mean,
                test_score_std=test_score_std,
                # compute the final score using function defined above:
                ranking_score=ranking_scorer(test_score_mean, test_score_std),
            )
            for search in self._searchers
            # we read and iterate over these 3 attributes from cv_results_:
            for params, test_score_mean, test_score_std in zip(
                search.cv_results_[ModelRanker.F_PARAMETERS],
                search.cv_results_[ModelRanker.F_MEAN_TEST_SCORE],
                search.cv_results_[ModelRanker.F_SD_TEST_SCORE],
            )
        ]

        # create ranking by assigning rank values and creating "RankedModel" types
        return ModelRanking(
            ranking=sorted(search_results, key=lambda result: -result.ranking_score)
        )

    @property
    def pipeline(self) -> Pipeline:
        """
        Property of ModelRanker

        :return: the complete scikit-learn pipeline
        """
        return self._pipeline

    def searchers(self) -> Iterable[GridSearchCV]:
        return iter(self._searchers)

    def models(self) -> Iterable[Model]:
        return iter(self._models)


class ModelRanking:
    BEST_MODEL_RANK = 0

    """
    Utility class that wraps a list of RankedModel
    """

    def __init__(self, ranking: List[RankedModel]):
        """
        Utility class that wraps a list of RankedModel

        :param ranking: the list of RankedModel instances this ranking is based on
        """
        self.__ranking = ranking

    def model(self, rank: int) -> RankedModel:
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
                rank + 1,
                ranked_model.estimator.__class__.__name__,
                ranked_model.ranking_score,
                ranked_model.parameters,
            )
            for rank, ranked_model in enumerate(self.__ranking[:limit])
        ]

        name_width = max([len(row[1]) for row in rows])

        return "\n".join(
            [
                f" Rank {row[0]:2d}: "
                f"{row[1]:>{name_width}s}, "
                f"Score={row[2]:.2e}, "
                f"Params={row[3]}"
                for row in rows
            ]
        )

    def __len__(self) -> int:
        return len(self.__ranking)

    def __str__(self) -> str:
        return self.summary_report()

    def __iter__(self) -> Iterable[RankedModel]:
        return iter(self.__ranking)
