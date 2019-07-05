import re
from collections import defaultdict
from itertools import chain
from typing import *

import numpy as np
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from yieldengine import Sample
from yieldengine.model import Model

ParameterGrid = Dict[str, Sequence[Any]]


class ModelGrid:
    def __init__(
        self,
        model: Model,
        estimator_parameters: ParameterGrid,
        preprocessing_parameters: Optional[ParameterGrid] = None,
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
    def estimator_parameters(self) -> ParameterGrid:
        return self._estimator_parameters

    @property
    def preprocessing_parameters(self) -> Optional[ParameterGrid]:
        return self._preprocessing_parameters

    @property
    def parameters(self) -> ParameterGrid:
        return self._grid


class ModelScoring:
    def __init__(self, split_scores: Iterable[float]):
        self.split_scores = np.array(split_scores)

    def mean(self) -> float:
        return self.split_scores.mean()

    def std(self) -> float:
        return self.split_scores.std()


class ModelEvaluation(NamedTuple):
    model: Model
    parameters: Mapping[str, Any]
    scoring: Mapping[str, ModelScoring]
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
    F_TEST_SCORE = "test_score"

    def __init__(
        self,
        grids: Iterable[ModelGrid],
        cv: Optional[BaseCrossValidator] = None,
        scoring: Union[
            str,
            Callable[[float, float], float],
            List[str],
            Tuple[str],
            Dict[str, Callable[[float, float], float]],
            None,
        ] = None,
    ) -> None:
        self._grids = list(grids)
        self._cv = cv
        self._scoring = scoring

    @staticmethod
    def default_ranking_scorer(scoring: ModelScoring) -> float:
        """
        The default scoring function to evaluate on top of GridSearchCV test scores,
        given by :code:`GridSearchCV.cv_results_`.

        Its output is used for ranking globally across the model zoo.

        :param scoring: the model scoring dictionary with entries for all scorers
        :return: score to be used for model ranking
        """
        return scoring.mean() - 2 * scoring.std()

    def run(
        self,
        sample: Sample,
        ranking_scorer: Callable[[float, float], float] = None,
        ranking_metric: str = "test_score",
        n_jobs: Optional[int] = None,
        pre_dispatch: str = "2*n_jobs",
    ) -> Sequence[ModelEvaluation]:
        """
        Execute the pipeline with the given sample and return the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models,
        taking mean and standard deviation of the ranking scores_for_split and returning the
        overall ranking score (default: ModelRanking.default_ranking_scorer)
        :param ranking_metric: the scoring to be used for model ranking, given as a name to be used to look up the right
               ModelScoring object in the ModelEvaluation.scoring dictionary (default: 'test_score').
        :param n_jobs: number of threads to use (default: one)
        :param pre_dispatch: maximum number of the data to make (default: `"2*n_jobs"`)

        :return the created model ranking of type :code:`ModelRanking`

        """

        # construct searchers
        searchers: List[Tuple[GridSearchCV, ModelGrid]] = [
            (
                GridSearchCV(
                    estimator=grid.model.pipeline,
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

        #
        # consolidate results of all searchers into "results"
        #

        def _scoring(
            cv_results: Mapping[str, Sequence[float]]
        ) -> List[Dict[str, ModelScoring]]:
            """
            helper function;  for each model in the grid returns a tuple of test scores_for_split across all splits.
            The length of the tuple is equal to the number of splits that were tested
            The test scores_for_split are sorted in the order the splits were tested
            :param cv_results: the GridSearchCV object's results dictionary
            :return: a list of test scores per scored model; each list entry maps score types to a list of scores per
                     split
            """

            # the splits are stored in the cv_results using keys 'split0...' thru 'split<nn>...'
            # match these dictionary keys in cv_results; ignore all other keys
            matches_for_split_x_metric: List[Tuple[str, Match]] = [
                (
                    key,
                    re.fullmatch(
                        pattern=r"split(\d+)_((train|test)_[a-zA-Z0-9]+)", string=key
                    ),
                )
                for key in cv_results.keys()
            ]

            # extract the integer indices from the matched results keys
            # create tuples (metric, split_index, scores_per_model_for_split), e.g., ('test_r2', 0, [0.34, 0.23, ...])
            metric_x_split_index_x_scores_per_model: List[
                Tuple[str, int, Sequence[float]]
            ] = sorted(
                (
                    (match.group(2), int(match.group(1)), cv_results[key])
                    for key, match in matches_for_split_x_metric
                    if match is not None
                ),
                key=lambda x: x[
                    1
                ],  # sort by split_id so we can later collect scores in the correct sequence
            )

            # Group results per model, result is a list where each item contains the scoring for one model.
            # Each scoring is a dictionary, mapping each metric to a list of scores for the different splits.
            n_models = len(cv_results[ModelRanker.F_PARAMETERS])

            scores_per_model_per_metric_per_split: List[Dict[str, List[float]]] = [
                defaultdict(list) for _ in range(n_models)
            ]

            for (
                metric,
                split_ix,
                split_score_per_model,
            ) in metric_x_split_index_x_scores_per_model:
                for model_ix, split_score in enumerate(split_score_per_model):
                    scores_per_model_per_metric_per_split[model_ix][metric].append(
                        split_score
                    )

            return [
                {
                    metric: ModelScoring(split_scores=scores_per_split)
                    for metric, scores_per_split in scores_per_metric_per_split.items()
                }
                for scores_per_metric_per_split in scores_per_model_per_metric_per_split
            ]

        scorings = [
            ModelEvaluation(
                model=grid.model.clone(parameters=params),
                parameters=params,
                scoring=scoring,
                # compute the final score using function defined above:
                ranking_score=ranking_scorer(scoring[ranking_metric]),
            )
            for searcher, grid in searchers
            # we read and iterate over these 3 attributes from cv_results_:
            for params, scoring in zip(
                searcher.cv_results_[ModelRanker.F_PARAMETERS],
                _scoring(searcher.cv_results_),
            )
        ]

        # create ranking by assigning rank values and creating "RankedModel" types
        return sorted(scorings, key=lambda scoring: scoring.ranking_score, reverse=True)


def summary_report(ranking: Sequence[ModelEvaluation]) -> str:
    """
    :return: a summary string of the model ranking
    """

    def _model_name(evaluation: ModelEvaluation) -> str:
        return evaluation.model.estimator.__class__.__name__

    def _parameters(params: Mapping[str, Iterable[Any]]) -> str:
        return ",".join(
            [
                f"{param_name}={param_value}"
                for param_name, param_value in params.items()
            ]
        )

    def _score_summary(scoring_dict: Mapping[str, ModelScoring]) -> str:
        return ", ".join(
            [
                f"{score}_mean={scoring.mean():9.3g}, "
                f"{score}_std={scoring.std():9.3g}, "
                for score, scoring in sorted(
                    scoring_dict.items(), key=lambda pair: pair[0]
                )
            ]
        )

    name_width = max([len(_model_name(ranked_model)) for ranked_model in ranking])

    return "\n".join(
        [
            f"Rank {rank + 1:2d}: "
            f"{_model_name(evaluation):>{name_width}s}, "
            f"Score={evaluation.ranking_score:9.3g}, "
            f"{_score_summary(evaluation.scoring)}, "
            f"Parameters={{{_parameters(evaluation.parameters)}}}"
            "\n"
            for rank, evaluation in enumerate(ranking)
        ]
    )
