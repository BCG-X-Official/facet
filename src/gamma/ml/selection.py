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
import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import *

import numpy as np
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import ClassifierCrossfit, LearnerCrossfit, RegressorCrossfit
from gamma.sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "ParameterGrid",
    "Scoring",
    "LearnerEvaluation",
    "LearnerRanker",
    "RegressorRanker",
    "ClassifierRanker",
]

#
# Type variables
#

_T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)
_T_RegressorPipelineDF = TypeVar("_T_RegressorPipelineDF", bound=RegressorPipelineDF)
_T_ClassifierPipelineDF = TypeVar("_T_ClassifierPipelineDF", bound=ClassifierPipelineDF)

_T_Crossfit = TypeVar("T_PredictionCV", bound=LearnerCrossfit[_T_LearnerPipelineDF])

# noinspection PyShadowingBuiltins
_T = TypeVar("_T")

#
# Class definitions
#


class ParameterGrid(Generic[_T_LearnerPipelineDF]):
    """
    A grid of hyper-parameters for pipeline tuning.

    :param pipeline: the :class:`ModelPipelineDF` to which the hyper-parameters will \
        be applied
    :param learner_parameters: the hyper-parameter grid in which to search for the \
        optimal parameter values for the pipeline's final estimator
    :param preprocessing_parameters: the hyper-parameter grid in which to search for \
        the optimal parameter values for the pipeline's preprocessing pipeline \
        (optional)
    """

    def __init__(
        self,
        pipeline: _T_LearnerPipelineDF,
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
            parameters=learner_parameters, prefix=pipeline.final_estimator_name
        )
        if preprocessing_parameters is not None:
            grid_parameters = chain(
                grid_parameters,
                _prefix_parameter_names(
                    parameters=preprocessing_parameters,
                    prefix=pipeline.preprocessing_name,
                ),
            )

        self._grid = dict(grid_parameters)

    @property
    def pipeline(self) -> _T_LearnerPipelineDF:
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


class LearnerEvaluation(Generic[_T_LearnerPipelineDF]):
    """
    LearnerEvaluation result for a specific parametrisation of a
    :class:`~gamma.sklearndf.pipeline.LearnerPipelineDF`, determined by a
    :class:`~gamma.ml.selection.LearnerRanker`

    :param pipeline: the unfitted :class:`~gamma.ml.LearnerPipelineDF`
    :param parameters: the hyper-parameters selected for the learner during grid \
        search, as a mapping of parameter names to parameter values
    :param scoring: maps score names to :class:`~gamma.ml.Scoring` instances
    :param ranking_score: overall score determined by the 's ranking \
        metric, used for ranking all crossfit
    """

    def __init__(
        self,
        pipeline: _T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scoring: Mapping[str, Scoring],
        ranking_score: float,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.parameters = parameters
        self.scoring = scoring
        self.ranking_score = ranking_score


class LearnerRanker(
    ParallelizableMixin, ABC, Generic[_T_LearnerPipelineDF, _T_Crossfit]
):
    """
    Rank different parametrisations of one or more learners using cross-validation.

    Given a list of :class:`~gamma.ml.ParameterGrid`, a cross-validation splitter and a
    scoring function, performs a grid search to find the pipeline and
    hyper-parameters with the best score across all cross-validation splits.

    Ranking is performed lazily when first invoking any method that depends on the
    result of the ranking, and then cached for subsequent calls.

    :param grid: :class:`~gamma.ml.ParameterGrid` to be ranked (either single grid or \
        an iterable of multiple grids)
    :param cv: a cross validator (e.g., \
        :class:`~gamma.ml.validation.BootstrapCV`)
    :param scoring: a scorer to use when doing CV within GridSearch, defaults to \
        :meth:`.default_ranking_scorer`
    :param ranking_scorer: scoring function used for ranking across crossfit, \
        taking mean and standard deviation of the ranking scores_for_split and \
        returning the overall ranking score (default: :meth:`.default_ranking_scorer`)
    :param ranking_metric: the scoring to be used for pipeline ranking, \
        given as a name to be used to look up the right Scoring object in the \
        LearnerEvaluation.scoring dictionary (default: 'test_score').
    :param n_jobs: number of jobs to use in parallel; \
        if `None`, use joblib default (default: `None`).
    :param shared_memory: if `True` use threads in the parallel runs. If `False` \
        use multiprocessing (default: `False`).
    :param pre_dispatch: number of batches to pre-dispatch; \
        if `None`, use joblib default (default: `None`).
    :param verbose: verbosity level used in the parallel computation; \
        if `None`, use joblib default (default: `None`).
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
        "_ranking",
    ]

    _COL_PARAMETERS = "params"

    def __init__(
        self,
        grid: Union[
            ParameterGrid[_T_LearnerPipelineDF],
            Iterable[ParameterGrid[_T_LearnerPipelineDF]],
        ],
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
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self._grids = list(grid) if isinstance(grid, Iterable) else [grid]
        self._cv = cv
        self._scoring = scoring
        self._ranking_scorer = (
            LearnerRanker.default_ranking_scorer
            if ranking_scorer is None
            else ranking_scorer
        )
        self._ranking_metric = ranking_metric

        # initialise state
        self._sample: Optional[Sample] = None
        self._fit_params: Optional[Dict[str, Any]] = None
        self._ranking: Optional[List[LearnerEvaluation]] = None

    @staticmethod
    def default_ranking_scorer(scoring: Scoring) -> float:
        """
        The default function to determine the pipeline's rank: ``mean - 2 * std``.

        Its output is used to rank different parametrizations of one or more learners.

        :param scoring: the :class:`Scoring` with validation scores for a given split
        :return: score to be used for pipeline ranking
        """
        return scoring.mean() - 2 * scoring.std()

    def fit(self: _T, sample: Sample, **fit_params) -> _T:
        """
        :param sample: sample with which to fit the candidate learners from the grid(s)
        :param fit_params: any fit parameters to pass on to the learner's fit method
        """
        cast(LearnerRanker, self)._rank_learners(sample=sample, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        """`True` if this ranker is fitted, `False` otherwise."""
        return self._sample is not None

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("expected ranker to be fitted")

    def ranking(self) -> List[LearnerEvaluation[_T_LearnerPipelineDF]]:
        """
        :return a ranking of all learners that were evaluated based on the parameter
        grids passed to this ranker, in descending order of the ranking score.
        """
        self._ensure_fitted()
        return self._ranking.copy()

    @property
    def best_model(self) -> _T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample
        """
        return self._best_pipeline().fit(X=self._sample.features, y=self._sample.target)

    def best_model_crossfit(
        self,
        cv: Optional[BaseCrossValidator] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[str] = None,
        verbose: Optional[int] = None,
    ) -> _T_Crossfit:
        """
        The crossfit for the best model, fitted with the same sample and fit
        parameters used to fit this ranker.

        :param cv: the cross-validator to use for generating the crossfit (default: \
            use this ranker's cross-validator)
        :param n_jobs: number of threads to use \
            (default: inherit this ranker's setting)
        :param shared_memory: whether to use threading with shared memory \
            (default: inherit this ranker's setting)
        :param pre_dispatch: maximum number of the data to make \
            (default: inherit this ranker's setting)
        :param verbose: verbosity of parallel processing \
            (default: inherit this ranker's setting)
        """

        return self._make_crossfit(
            pipeline=self._best_pipeline(),
            cv=self._cv if cv is None else cv,
            n_jobs=self.n_jobs if n_jobs is None else n_jobs,
            shared_memory=self.shared_memory
            if shared_memory is None
            else shared_memory,
            pre_dispatch=self.pre_dispatch if pre_dispatch is None else pre_dispatch,
            verbose=self.verbose if verbose is None else verbose,
        ).fit(sample=self._sample, **self._fit_params)

    def summary_report(self, max_learners: Optional[int] = None) -> str:
        """
        Return a human-readable report of learner validation results, sorted by
        ranking score in descending order.

        :param max_learners: maximum number of learners to include in the report \
            (optional)

        :return: a summary string of the pipeline ranking
        """

        self._ensure_fitted()

        def _model_name(evaluation: LearnerEvaluation) -> str:
            return type(evaluation.pipeline.final_estimator).__name__

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

    def _best_pipeline(self) -> _T_LearnerPipelineDF:
        # return the unfitted model with the best parametrisation
        self._ensure_fitted()
        return self._ranking[0].pipeline

    @abstractmethod
    def _make_crossfit(
        self,
        pipeline: _T_LearnerPipelineDF,
        cv: BaseCrossValidator,
        n_jobs: int,
        shared_memory: bool,
        pre_dispatch: str,
        verbose: int,
    ) -> _T_Crossfit:
        pass

    def _rank_learners(self, sample: Sample, **fit_params) -> None:

        if len(fit_params) > 0:
            log.warning(
                "Ignoting arg fit_params: current ranker implementation uses "
                "GridSearchCV which does not support fit_params"
            )

        ranking_scorer = self._ranking_scorer

        # construct searchers
        searchers: List[Tuple[GridSearchCV, ParameterGrid]] = [
            (
                GridSearchCV(
                    estimator=grid.pipeline,
                    param_grid=grid.parameters,
                    scoring=self._scoring,
                    n_jobs=self.n_jobs,
                    iid=False,
                    refit=False,
                    cv=self._cv,
                    verbose=self.verbose,
                    pre_dispatch=self.pre_dispatch,
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
            :return: a list of test scores per scored pipeline; each list entry maps \
                score types (as str) to a :class:`Scoring` of scores per split. The \
                i-th element of this list is typically of the form \
                ``{'train_score': model_scoring1, 'test_score': model_scoring2,...}``
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
            LearnerEvaluation(
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

        self._sample = sample
        self._fit_params = fit_params
        self._ranking = ranking


class RegressorRanker(
    LearnerRanker[_T_RegressorPipelineDF, RegressorCrossfit[_T_RegressorPipelineDF]],
    Generic[_T_RegressorPipelineDF],
):
    def _make_crossfit(
        self,
        pipeline: _T_RegressorPipelineDF,
        cv: BaseCrossValidator,
        n_jobs: int,
        shared_memory: bool,
        pre_dispatch: str,
        verbose: int,
    ) -> RegressorCrossfit[_T_RegressorPipelineDF]:
        return RegressorCrossfit(
            base_estimator=pipeline,
            cv=cv,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )


class ClassifierRanker(
    LearnerRanker[_T_ClassifierPipelineDF, ClassifierCrossfit[_T_ClassifierPipelineDF]],
    Generic[_T_ClassifierPipelineDF],
):
    def _make_crossfit(
        self,
        pipeline: _T_ClassifierPipelineDF,
        cv,
        n_jobs,
        shared_memory,
        pre_dispatch,
        verbose,
    ) -> ClassifierCrossfit[_T_ClassifierPipelineDF]:
        return ClassifierCrossfit(
            base_estimator=pipeline,
            cv=cv,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
