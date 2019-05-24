from itertools import chain
from typing import *

from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from yieldengine import Sample
from yieldengine.model import Model


class ModelGrid:
    def __init__(
        self,
        model: Model,
        estimator_parameters: Dict[str, Any],
        preprocessing_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = model
        self._estimator_parameters = estimator_parameters
        self._preprocessing_parameters = preprocessing_parameters

        def _prefix_parameter_names(
            parameters: Dict[str, Any], prefix: str
        ) -> List[Tuple[str, Any]]:
            return [
                (f"{prefix}__{param}", value) for param, value in parameters.items()
            ]

        grid_parameters: Iterable[Tuple[str, Any]] = _prefix_parameter_names(
            parameters=estimator_parameters, prefix=Model.STEP_ESTIMATOR
        )
        if preprocessing_parameters is not None:
            grid_parameters = chain(
                grid_parameters,
                _prefix_parameter_names(
                    parameters=preprocessing_parameters, prefix=Model.STEP_PREPROCESSING
                ),
            )

        self._grid = dict(grid_parameters)

    @property
    def model(self) -> Model:
        return self._model

    @property
    def estimator_parameters(self) -> Dict[str, Any]:
        return self._estimator_parameters

    @property
    def preprocessing_parameters(self) -> Optional[Dict[str, Any]]:
        return self._preprocessing_parameters

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._grid


class ModelEvaluation(NamedTuple):
    model: Model
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

    :param grids: list of model grids to be ranked
    :param cv: a cross validation object (i.e. CircularCrossValidator)
    :param scoring: a scorer to use when doing CV within GridSearch
    """

    __slots__ = ["_grids", "_scoring", "_cv", "_searchers", "_pipeline"]

    F_PARAMETERS = "params"
    F_MEAN_TEST_SCORE = "mean_test_score"
    F_SD_TEST_SCORE = "std_test_score"

    def __init__(
        self,
        grids: Iterable[ModelGrid],
        cv: Optional[BaseCrossValidator] = None,
        scoring=None,
    ) -> None:
        self._grids = list(grids)
        self._cv = cv
        self._scoring = scoring

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
        self,
        sample: Sample,
        ranking_scorer: Callable[[float, float], float] = None,
        n_jobs: Optional[int] = None,
        pre_dispatch="2*n_jobs",
    ) -> Sequence[ModelEvaluation]:
        """
        Execute the pipeline with the given sample and return the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models,
        taking mean and standard deviation of the ranking scores and returning the
        overall ranking score (default: ModelRanking.default_ranking_scorer)
        :param n_jobs: number of threads to use (default: one)
        :param pre_dispatch: maximum number of the data to make (default: `"2*n_jobs"`)

        :return the created model ranking of type :code:`ModelRanking`

        """

        # construct searchers
        searchers: List[Tuple[GridSearchCV, ModelGrid]] = [
            (
                GridSearchCV(
                    estimator=grid.model.pipeline(),
                    cv=self._cv,
                    param_grid=grid.parameters,
                    scoring=self._scoring,
                    return_train_score=False,
                    n_jobs=n_jobs,
                    pre_dispatch=pre_dispatch,
                    refit=False,
                ),
                grid,
            )
            for grid in self._grids
        ]

        for searcher, _ in searchers:
            searcher.fit(X=sample.features, y=sample.target)

        if ranking_scorer is None:
            ranking_scorer = ModelRanker.default_ranking_scorer

        # consolidate results of all searchers into "results"
        scorings = [
            ModelEvaluation(
                model=grid.model.clone(parameters=params),
                test_score_mean=test_score_mean,
                test_score_std=test_score_std,
                # compute the final score using function defined above:
                ranking_score=ranking_scorer(test_score_mean, test_score_std),
            )
            for search, grid in searchers
            # we read and iterate over these 3 attributes from cv_results_:
            for params, test_score_mean, test_score_std in zip(
                search.cv_results_[ModelRanker.F_PARAMETERS],
                search.cv_results_[ModelRanker.F_MEAN_TEST_SCORE],
                search.cv_results_[ModelRanker.F_SD_TEST_SCORE],
            )
        ]

        # create ranking by assigning rank values and creating "RankedModel" types
        return sorted(scorings, key=lambda scoring: scoring.ranking_score, reverse=True)


def summary_report(ranking: Sequence[ModelEvaluation]) -> str:
    """
    :return: a summary string of the model ranking
    """

    def _model_name(ranked_model: ModelEvaluation) -> str:
        return ranked_model.model.estimator.__class__.__name__

    name_width = max([len(_model_name(ranked_model)) for ranked_model in ranking])

    def parameters(params: Dict[str, Iterable[Any]]) -> str:
        return "\n    ".join(
            [
                f"{param_name}={param_value}"
                for param_name, param_value in params.items()
            ]
        )

    return "\n".join(
        [
            f"Rank {rank + 1:2d}: "
            f"{_model_name(ranked_model):>{name_width}s}, "
            f"Score={ranked_model.ranking_score:+.2e}, "
            f"Test mean={ranked_model.test_score_mean:+.2e}, "
            f"Test std={ranked_model.test_score_std:+.2e}"
            "\nEstimator parameters:"
            f"\n    {parameters(ranked_model.model.estimator.get_params())}"
            + (
                ""
                if ranked_model.model.preprocessing is None
                else "\nPreprocessing parameters:"
                f"\n    {parameters(ranked_model.model.preprocessing.get_params())}"
            )
            + "\n"
            for rank, ranked_model in enumerate(ranking)
        ]
    )
