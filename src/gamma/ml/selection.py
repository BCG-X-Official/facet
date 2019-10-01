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

:class:`LearnerRanker` selects the best pipeline and parametrisation based on the
pipeline and hyperparameter choices provided as a list of :class:`ModelGrid`.
"""
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import *

import copy
import numpy as np
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from gamma.ml import Sample
from gamma.ml.predictioncv import (
    ClassifierPredictionCV,
    PredictionCV,
    RegressorPredictionCV,
)
from gamma.sklearndf import ClassifierDF, RegressorDF
from gamma.sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

__all__ = ["ParameterGrid", "Scoring", "Validation", "LearnerRanker"]

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)
T_PredictionCV = TypeVar("T_PredictionCV", bound=PredictionCV[T_LearnerPipelineDF])
_T_FinalRegressorDF = TypeVar("_T_FinalRegressorDF", bound=RegressorDF)
_T_FinalClassifierDF = TypeVar("_T_FinalClassifierDF", bound=ClassifierDF)
_T = TypeVar("_T")


class ParameterGrid(Generic[T_LearnerPipelineDF]):
    """
    A grid of hyper-parameters for pipeline tuning.

    :param pipeline: the :class:`ModelPipelineDF` to which the hyper-parameters will \
        be applied
    :param learner_parameters: the hyper-parameter grid in which to search for the \
        optimal parameter values for the pipeline's final estimator
    :param preprocessing_parameters: the hyper-parameter grid in which to search for \
        the optimal parameter values for the pipeline's preprocessing pipeline (optional)
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        learner_parameters: Dict[str, Sequence[Any]],
        preprocessing_parameters: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self._pipeline = pipeline
        self._learner_parameters = learner_parameters
        self._preprocessing_parameters = preprocessing_parameters

        def _prefix_parameter_names(
            parameters: Dict[str, Any], prefix: str
        ) -> List[Tuple[str, Any]]:
            return [
                (f"{prefix}__{param}", value) for param, value in parameters.items()
            ]

        grid_parameters: Iterable[Tuple[str, Any]] = _prefix_parameter_names(
            parameters=learner_parameters, prefix=pipeline.final_estimator_param_
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
    def pipeline(self) -> T_LearnerPipelineDF:
        """
        The :class:`~gamma.ml.EstimatorPipelineDF` for which to optimise the
        parameters.
        """
        return self._pipeline

    @property
    def learner_parameters(self) -> Dict[str, Sequence[Any]]:
        """The parameter grid for the estimator."""
        return self._learner_parameters

    @property
    def preprocessing_parameters(self) -> Optional[Dict[str, Sequence[Any]]]:
        """The parameter grid for the preprocessor."""
        return self._preprocessing_parameters

    @property
    def parameters(self) -> Dict[str, Sequence[Any]]:
        """The parameter grid for the pipeline representing the entire pipeline."""
        return self._grid


class Scoring:
    """"
    Basic statistics on the scoring across all cross validation splits of a pipeline.

    :param split_scores: scores of all cross validation splits for a pipeline
    """

    def __init__(self, split_scores: Iterable[float]):
        self._split_scores = np.array(split_scores)

    def __getitem__(self, item: Union[int, slice]) -> Union[float, np.ndarray]:
        return self._split_scores[item]

    def mean(self) -> float:
        """:return: mean of the split scores"""
        return self._split_scores.mean()

    def std(self) -> float:
        """:return: standard deviation of the split scores"""
        return self._split_scores.std()


class Validation(Generic[T_LearnerPipelineDF]):
    """
    Validation result for a specific parametrisation of a
    :class:`~gamma.sklearndf.pipeline.LearnerPipelineDF`, determined by a
    :class:`~gamma.ml.selection.`

    :param pipeline: the (unfitted) :class:`~gamma.ml.LearnerPipelineDF`
    :param parameters: the hyper-parameters selected for the learner during grid \
        search, as a mapping of parameter names to parameter values
    :param scoring: maps score names to :class:`~gamma.ml.Scoring` instances
    :param ranking_score: overall score determined by the 's ranking \
        metric, used for ranking all predictions
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scoring: Mapping[str, Scoring],
        ranking_score: float,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.parameters = parameters
        self.scoring = scoring
        self.ranking_score = ranking_score


class LearnerRanker(Generic[T_LearnerPipelineDF, T_PredictionCV], ABC):
    """
    Rank a list of predictions using a cross-validation.

    Given a list of :class:`~gamma.ml.ParameterGrid`, a cross-validation splitter and a
    scoring function, performs a grid search to find the pipeline and
    hyper-parameters with the best score across all cross-validation splits.

    Ranking is performed lazily when first invoking any method that depends on the
    result of the ranking, and then cached for subsequent calls.

    :param grid: :class:`~gamma.ml.ParameterGrid` to be ranked (either single grid or \
        an iterable of multiple grids)
    :param sample: sample with which to fit the candidate learners from the grid(s)
    :param cv: a cross validator (e.g., \
        :class:`~gamma.ml.validation.BootstrapCV`)
    :param scoring: a scorer to use when doing CV within GridSearch, defaults to \
        :meth:`.default_ranking_scorer`
    :param ranking_scorer: scoring function used for ranking across predictions, \
        taking mean and standard deviation of the ranking scores_for_split and \
        returning the overall ranking score (default: :meth:`.default_ranking_scorer`)
    :param ranking_metric: the scoring to be used for pipeline ranking, \
        given as a name to be used to look up the right Scoring object in the \
        Validation.scoring dictionary (default: 'test_score').
    :param n_jobs: number of threads to use (default: one)
    :param shared_memory: whether to use threading with shared memory (default: `False`)
    :param pre_dispatch: maximum number of the data to make (default: `"2*n_jobs"`)
    :param verbose: verbosity of parallel processing (default: 0)
    """

    __slots__ = [
        "_grids",
        "_sample",
        "_scoring",
        "_cv",
        "_searchers",
        "_pipeline",
        "_ranking_scorer",
        "_ranking_metric",
        "_n_jobs",
        "_shared_memory",
        "_pre_dispatch",
        "_verbose",
        "_ranking",
    ]

    _COL_PARAMETERS = "params"

    def __init__(
        self,
        grid: Union[
            ParameterGrid[T_LearnerPipelineDF],
            Iterable[ParameterGrid[T_LearnerPipelineDF]],
        ],
        sample: Sample,
        cv: Optional[BaseCrossValidator],
        scoring: Union[
            str,
            Callable[[float, float], float],
            List[str],
            Tuple[str],
            Dict[str, Callable[[float, float], float]],
            None,
        ] = None,
        ranking_scorer: Callable[[float, float], float] = None,
        ranking_metric: str = "test_score",
        n_jobs: Optional[int] = None,
        shared_memory: bool = False,
        pre_dispatch: str = "2*n_jobs",
        verbose: int = 0,
    ) -> None:
        self._grids = [grid] if isinstance(grid, ParameterGrid) else list(grid)
        self._sample = sample
        self._cv = cv
        self._scoring = scoring
        self._ranking_scorer = (
            LearnerRanker.default_ranking_scorer
            if ranking_scorer is None
            else ranking_scorer
        )
        self._ranking_metric = ranking_metric
        self._n_jobs = n_jobs
        self._shared_memory = shared_memory
        self._pre_dispatch = pre_dispatch
        self._verbose = verbose

        # initialise state
        self._ranking: Optional[List[Validation]] = None

    @staticmethod
    def default_ranking_scorer(scoring: Scoring) -> float:
        """
        The default function to determine the pipeline's rank: ``mean - 2 * std``.

        Its output is used to rank different parametrizations of one or more learners.

        :param scoring: the :class:`Scoring` with validation scores for a given split
        :return: score to be used for pipeline ranking
        """
        return scoring.mean() - 2 * scoring.std()

    def ranking(self) -> List[Validation[T_LearnerPipelineDF]]:
        """
        :return a ranking of all learners that were evaluated based on the parameter
        grids passed to this ranker, in descending order of the ranking score.
        """
        self._rank_learners()

        return self._ranking.copy()

    def best_model(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample
        """
        return self._best_pipeline().fit(X=self._sample.features, y=self._sample.target)

    def best_model_predictions(self) -> T_PredictionCV:
        """
        The cross-validated predictions for the best model
        """
        return self._fit_and_predict_model(self._best_pipeline())

    def _best_pipeline(self) -> T_LearnerPipelineDF:
        # return the unfitted model with the best parametrisation
        self._rank_learners()
        return self._ranking[0].pipeline

    def summary_report(self, max_learners: Optional[int] = None) -> str:
        """
        Return a human-readable report of learner validation results, sorted by
        ranking score in descending order.

        :param max_learners: maximum number of learners to include in the report \
            (optional)

        :return: a summary string of the pipeline ranking
        """

        def _model_name(evaluation: Validation) -> str:
            return evaluation.pipeline.final_estimator_.__class__.__name__

        def _parameters(params: Mapping[str, Iterable[Any]]) -> str:
            return ",".join(
                [
                    f"{param_name}={param_value}"
                    for param_name, param_value in params.items()
                ]
            )

        def _score_summary(scoring_dict: Mapping[str, Scoring]) -> str:
            return ", ".join(
                [
                    f"{score}_mean={scoring.mean():9.3g}, "
                    f"{score}_std={scoring.std():9.3g}, "
                    for score, scoring in sorted(
                        scoring_dict.items(), key=lambda pair: pair[0]
                    )
                ]
            )

        self._rank_learners()

        ranking = self._ranking[:max_learners] if max_learners else self._ranking

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

    def copy_with_sample(self: _T, sample: Sample) -> _T:
        clone = copy.copy(self)
        clone._sample = sample
        clone._ranking = None
        return clone

    @abstractmethod
    def _fit_and_predict_model(self, pipeline: T_LearnerPipelineDF) -> T_PredictionCV:
        pass

    def _rank_learners(self) -> None:

        if self._ranking is not None:
            return

        sample = self._sample
        ranking_scorer = self._ranking_scorer
        # construct searchers
        searchers: List[Tuple[GridSearchCV, ParameterGrid]] = [
            (
                GridSearchCV(
                    estimator=grid.pipeline,
                    param_grid=grid.parameters,
                    scoring=self._scoring,
                    n_jobs=self._n_jobs,
                    iid=False,
                    refit=False,
                    cv=self._cv,
                    verbose=self._verbose,
                    pre_dispatch=self._pre_dispatch,
                    return_train_score=False,
                ),
                grid,
            )
            for grid in self._grids
        ]

        for searcher, _ in searchers:
            searcher.fit(X=sample.features, y=sample.target)

        #
        # consolidate results of all searchers into "results"
        #

        def _scoring(
            cv_results: Mapping[str, Sequence[float]]
        ) -> List[Dict[str, Scoring]]:
            """
            Convert ``cv_results_`` into a mapping with :class:`Scoring` values.

            Helper function;  for each pipeline in the grid returns a tuple of test
            scores_for_split across all splits.
            The length of the tuple is equal to the number of splits that were tested
            The test scores_for_split are sorted in the order the splits were tested.

            :param cv_results: a :attr:`sklearn.GridSearchCV.cv_results_` attribute
            :return: a list of test scores per scored pipeline; each list entry maps score
              types (as str) to a :class:`Scoring` of scores per split. The i-th
              element of this
              list is
              typically of the form ``{'train_score': model_scoring1, 'test_score':
              model_scoring2,...}``
            """

            # the splits are stored in the cv_results using keys 'split0...'
            # through 'split<nn>...'
            # match these dictionary keys in cv_results; ignore all other keys
            matches_for_split_x_metric: List[Tuple[str, Match]] = [
                (key, re.fullmatch(r"split(\d+)_((train|test)_[a-zA-Z0-9]+)", key))
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

            # Group results per pipeline, result is a list where each item contains the
            # scoring for one pipeline. Each scoring is a dictionary, mapping each
            # metric to a list of scores for the different splits.
            n_models = len(cv_results[LearnerRanker._COL_PARAMETERS])

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
            # a_j (resp. b_j) is the train (resp. test) score for pipeline i in split j

            return [
                {
                    metric: Scoring(split_scores=scores_per_split)
                    for metric, scores_per_split in scores_per_metric_per_split.items()
                }
                for scores_per_metric_per_split in scores_per_model_per_metric_per_split
            ]

        ranking_metric = self._ranking_metric
        ranking = [
            Validation(
                pipeline=grid.pipeline.clone().set_params(**params),
                parameters=params,
                scoring=scoring,
                # compute the final score using the function defined above:
                ranking_score=ranking_scorer(scoring[ranking_metric]),
            )
            for searcher, grid in searchers
            # we read and iterate over these 3 attributes from cv_results_:
            for params, scoring in zip(
                searcher.cv_results_[LearnerRanker._COL_PARAMETERS],
                _scoring(searcher.cv_results_),
            )
        ]

        ranking.sort(key=lambda validation: validation.ranking_score, reverse=True)

        self._ranking = ranking


class RegressorRanker(
    LearnerRanker[
        RegressorPipelineDF[_T_FinalRegressorDF],
        RegressorPredictionCV[RegressorPipelineDF[_T_FinalRegressorDF]],
    ],
    Generic[_T_FinalRegressorDF],
):
    def _fit_and_predict_model(
        self, pipeline: RegressorPipelineDF[_T_FinalRegressorDF]
    ) -> RegressorPredictionCV[RegressorPipelineDF[_T_FinalRegressorDF]]:
        return RegressorPredictionCV(
            pipeline=pipeline,
            cv=self._cv,
            sample=self._sample,
            n_jobs=self._n_jobs,
            shared_memory=self._shared_memory,
            verbose=self._verbose,
        )


class ClassifierRanker(
    LearnerRanker[
        ClassifierPipelineDF[_T_FinalClassifierDF],
        ClassifierPredictionCV[ClassifierPipelineDF[_T_FinalClassifierDF]],
    ],
    Generic[_T_FinalClassifierDF],
):
    def _fit_and_predict_model(
        self, pipeline: ClassifierPipelineDF[_T_FinalClassifierDF]
    ) -> ClassifierPredictionCV[ClassifierPipelineDF[_T_FinalClassifierDF]]:
        return ClassifierPredictionCV(
            pipeline=pipeline,
            cv=self._cv,
            sample=self._sample,
            n_jobs=self._n_jobs,
            shared_memory=self._shared_memory,
            verbose=self._verbose,
        )
