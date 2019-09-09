#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
ModelPipelineDF selection and hyperparameter optimisation.

:class:`ParameterGrid` encapsulates a :class:`gamma.ml.ModelPipelineDF` and a grid of
hyperparameters.

:class:`ModelRanker` selects the best model and parametrisation based on the
model and hyperparameter choices provided as a list of :class:`ModelGrid`.
"""
import re
from collections import defaultdict
from itertools import chain
from typing import *

import numpy as np
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from gamma.ml import Sample
from gamma.sklearndf.pipeline import EstimatorPipelineDF, PredictorPipelineDF


class ParameterGrid:
    """
    A grid of hyperparameters for model tuning.

    :param pipeline: the :class:`ModelPipelineDF` to which the hyperparameters will be
        applied
    :param estimator_parameters: the hyperparameter grid in which to search for the
        optimal parameter values for the pipeline's final estimator
    :param preprocessing_parameters: the hyperparameter grid in which to search for
        the optimal parameter values for the model's preprocessing pipeline (optional)
    """

    def __init__(
        self,
        pipeline: EstimatorPipelineDF,
        estimator_parameters: Dict[str, Sequence[Any]],
        preprocessing_parameters: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self._pipeline = pipeline
        self._estimator_parameters = estimator_parameters
        self._preprocessing_parameters = preprocessing_parameters

        def _prefix_parameter_names(
            parameters: Dict[str, Any], prefix: str
        ) -> List[Tuple[str, Any]]:
            return [
                (f"{prefix}__{param}", value) for param, value in parameters.items()
            ]

        grid_parameters: Iterable[Tuple[str, Any]] = _prefix_parameter_names(
            parameters=estimator_parameters, prefix=pipeline.final_estimator_param_
        )
        if preprocessing_parameters is not None:
            grid_parameters = chain(
                grid_parameters,
                _prefix_parameter_names(
                    parameters=preprocessing_parameters,
                    prefix=pipeline.preprocessing_param_,
                ),
            )

        self._grid = dict(grid_parameters)

    @property
    def pipeline(self) -> PredictorPipelineDF:
        """
        The :class:`~gamma.ml.PredictorPipelineDF` for which to optimise the
        parameters.
        """
        return self._pipeline

    @property
    def predictor_parameters(self) -> Dict[str, Sequence[Any]]:
        """The parameter grid for the estimator."""
        return self._estimator_parameters

    @property
    def preprocessing_parameters(self) -> Optional[Dict[str, Sequence[Any]]]:
        """The parameter grid for the preprocessor."""
        return self._preprocessing_parameters

    @property
    def parameters(self) -> Dict[str, Sequence[Any]]:
        """The parameter     grid for the pipeline representing the entire model."""
        return self._grid


class ModelScoring:
    """"
    Basic statistics on the scoring across all cross validation splits of a model.

    :param split_scores: scores of all cross validation splits for a model
    """

    def __init__(self, split_scores: Iterable[float]):
        self.split_scores = np.array(split_scores)

    def mean(self) -> float:
        """:return: mean of the split scores"""
        return self.split_scores.mean()

    def std(self) -> float:
        """:return: standard deviation of the split scores"""
        return self.split_scores.std()


class ModelEvaluation(NamedTuple):
    """
    Scoring evaluation for a fitted model.

    Has attributes:

    - model: the fitted  :class:`~gamma.ml.PredictorPipelineDF`
    - parameters: the hyperparameters selected for the model during grid
        search, as a mapping of parameter names to parameter values
    - scoring: scorings for the model based on the provided scorers;
        each scoring is applied across all splits. (e.g.,
        "train_score", "test_score", "train_r2", "test_r2")
    - ranking_score: overall model score determined by the model ranker's default
        scorer and ranking metric
    """

    model: PredictorPipelineDF
    parameters: Mapping[str, Any]
    scoring: Mapping[str, ModelScoring]
    ranking_score: float


class ModelRanker:
    """
    Rank a list of models using a cross-validation.

    Given a list of :class:`ModelGrid`, a cross-validation splitter and a scoring
    function, performs a grid search to find the best combination of model with
    hyperparameters for the given cross-validation splits and scoring function.

    :param grids: list of :class:`ModelGrid` to be ranked
    :param cv: a cross validator (i.e. \
        :class:`~gamma.ml.validation.CircularCrossValidator`)
    :param scoring: a scorer to use when doing CV within GridSearch
    """

    __slots__ = ["_grids", "_scoring", "_cv", "_searchers", "_pipeline"]

    F_PARAMETERS = "params"

    def __init__(
        self,
        grids: Iterable[ParameterGrid],
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
        The default scoring function: ``mean - 2*std``.

        Its output is used for ranking globally across the model zoo.

        :param scoring: the :class:`ModelScoring` containing scores for a given split
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
        Execute the pipeline for all models and compute the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models, \
        taking mean and standard deviation of the ranking scores_for_split and \
        returning the overall ranking score (default: \
        ModelRanking.default_ranking_scorer)
        :param ranking_metric: the scoring to be used for model ranking, \
        given as a name to be used to look up the right ModelScoring object in the \
        ModelEvaluation.scoring dictionary (default: 'test_score').
        :param n_jobs: number of threads to use (default: one)
        :param pre_dispatch: maximum number of the data to make (default: `"2*n_jobs"`)

        :return: the created model ranking
        """

        # construct searchers
        searchers: List[Tuple[GridSearchCV, ParameterGrid]] = [
            (
                GridSearchCV(
                    estimator=grid.pipeline,
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
            Convert ``cv_results_`` into a mapping with :class:`ModelScoring` values.

            Helper function;  for each model in the grid returns a tuple of test
            scores_for_split across all splits.
            The length of the tuple is equal to the number of splits that were tested
            The test scores_for_split are sorted in the order the splits were tested.

            :param cv_results: a :attr:`sklearn.GridSearchCV.cv_results_` attribute
            :return: a list of test scores per scored model; each list entry maps score
              types (as str) to a :class:`ModelScoring` of scores per split. The i-th
              element of this
              list is
              typically of the form ``{'train_score': model_scoring1, 'test_score':
              model_scoring2,...}``
            """

            # the splits are stored in the cv_results using keys 'split0...'
            # through 'split<nn>...'
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
            # create tuples (metric, split_index, scores_per_model_for_split),
            # e.g., ('test_r2', 0, [0.34, 0.23, ...])
            metric_x_split_index_x_scores_per_model: List[
                Tuple[str, int, Sequence[float]]
            ] = sorted(
                (
                    (match.group(2), int(match.group(1)), cv_results[key])
                    for key, match in matches_for_split_x_metric
                    if match is not None
                ),
                key=lambda x: x[1],  # sort by split_id so we can later collect scores
                # in the correct sequence
            )

            # Group results per model, result is a list where each item contains the
            # scoring for one model. Each scoring is a dictionary, mapping each
            # metric to a list of scores for the different splits.
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
            # Now in general, the i-th element of scores_per_model_per_metric_per_split
            # is a dict
            # {'train_score': [a_0,...,a_(n-1)], 'test_score': [b_0,..,b_(n-1)]} where
            # a_j (resp. b_j) is the train (resp. test) score for model i in split j

            return [
                {
                    metric: ModelScoring(split_scores=scores_per_split)
                    for metric, scores_per_split in scores_per_metric_per_split.items()
                }
                for scores_per_metric_per_split in scores_per_model_per_metric_per_split
            ]

        scorings = [
            ModelEvaluation(
                model=grid.pipeline.clone().set_params(**params),
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
        return sorted(
            scorings,
            key=lambda model_evaluation: model_evaluation.ranking_score,
            reverse=True,
        )


def summary_report(ranking: Sequence[ModelEvaluation]) -> str:
    """
    Return a human-readable report.

    :return: a summary string of the model ranking
    """

    def _model_name(evaluation: ModelEvaluation) -> str:
        return evaluation.model.final_estimator_.__class__.__name__

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
