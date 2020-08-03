"""
Core implementation of :mod:`gamma.ml.selection`
"""

import logging
import math
import operator
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import *

from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator

from gamma.common import to_tuple
from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import CrossfitScores, LearnerCrossfit
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF

log = logging.getLogger(__name__)

__all__ = ["LearnerScores", "LearnerGrid", "LearnerRanker"]

#
# Type variables
#

T_LearnerPipelineDF = TypeVar(
    "T_LearnerPipelineDF", RegressorPipelineDF, ClassifierPipelineDF
)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)

#
# Class definitions
#


class LearnerGrid(Sequence[Dict[str, Any]], Generic[T_LearnerPipelineDF]):
    """
    A grid of hyper-parameters for tuning a learner pipeline.

    :param pipeline: the :class:`.RegressorPipelineDF` or \
        :class:`.ClassifierPipelineDF` to which the hyper-parameters will be applied
    :param learner_parameters: the hyper-parameter grid in which to search for the \
        optimal parameter values for the pipeline's final estimator
    :param preprocessing_parameters: the hyper-parameter grid in which to search for \
        the optimal parameter values for the pipeline's preprocessing pipeline \
        (optional)
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        learner_parameters: Dict[str, Sequence],
        preprocessing_parameters: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        self.pipeline = pipeline

        def _prefix_parameter_names(
            parameters: Dict[str, Sequence], prefix: str
        ) -> Iterable[Tuple[str, Any]]:
            return (
                (f"{prefix}__{param}", values) for param, values in parameters.items()
            )

        grid_parameters: Iterable[Tuple[str, Sequence]] = _prefix_parameter_names(
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

        self._grid_parameters: List[Tuple[str, Sequence]] = list(grid_parameters)
        self._grid_dict: Dict[str, Sequence] = dict(self._grid_parameters)

    @property
    def parameters(self) -> Mapping[str, Sequence[Any]]:
        """The parameter grid for the pipeline representing the entire pipeline."""
        return MappingProxyType(self._grid_dict)

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        grid = self._grid_parameters
        params: List[Tuple[str, Any]] = [("", None) for _ in grid]

        def _iter_parameter(param_index: int):
            if param_index < 0:
                yield dict(params)
            else:
                name, values = grid[param_index]
                for value in values:
                    params[param_index] = (name, value)
                    yield from _iter_parameter(param_index=param_index - 1)

        yield from _iter_parameter(len(grid) - 1)

    def __getitem__(
        self, pos: Union[int, slice]
    ) -> Union[Dict[str, Sequence], Sequence[Dict[str, Sequence]]]:

        _len = len(self)

        def _get(i: int) -> Dict[str, Sequence]:
            assert i >= 0

            parameters = self._grid_parameters
            result: Dict[str, Sequence] = {}

            for name, values in parameters:
                n_values = len(values)
                result[name] = values[i % n_values]
                i //= n_values

            assert i == 0

            return result

        def _clip(i: int, i_max: int) -> int:
            if i < 0:
                return max(_len + i, 0)
            else:
                return min(i, i_max)

        if isinstance(pos, slice):
            print(pos)
            return [
                _get(i)
                for i in range(
                    _clip(pos.start or 0, _len - 1),
                    _clip(pos.stop or _len, _len),
                    pos.step or 1,
                )
            ]
        else:
            if pos < -_len or pos >= _len:
                raise ValueError(f"index out of bounds: {pos}")
            return _get(_len + pos if pos < 0 else pos)

    def __len__(self) -> int:
        return reduce(
            operator.mul,
            (
                len(values_for_parameter)
                for values_for_parameter in self._grid_dict.values()
            ),
        )


class LearnerScores(Generic[T_LearnerPipelineDF]):
    """
    A collection of scores for a specific parametrisation of a learner pipeline,
    generated by a :class:`.LearnerRanker`.

    :param pipeline: the unfitted learner pipeline
    :param parameters: the hyper-parameters selected for the learner during grid \
        search, as a mapping of parameter names to parameter values
    :param scores: the scores of all crossfits of the learner pipeline
        :class:`.CrossfitScores` instances
    :param ranking_score: overall score determined by the ranking \
        metric of the :class:`.LearnerRanker`, used for ranking the learners
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scores: CrossfitScores,
        ranking_score: float,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.parameters = parameters
        self.scores = scores
        self.ranking_score = ranking_score


class LearnerRanker(
    ParallelizableMixin, FittableMixin[Sample], Generic[T_LearnerPipelineDF]
):
    """
    Score and rank different parametrizations of one or more learners,
    using cross-validation.

    Other than scikit-learn's native :class:`.GridSearchCV`, a learner ranker can
    run a simultaneous grid search across multiple alternative learner pipelines,
    allowing not only to optimize hyper-parameters, but also the choice of the
    learner algorithm.
    """

    def __init__(
        self,
        grid: Union[
            LearnerGrid[T_LearnerPipelineDF], Iterable[LearnerGrid[T_LearnerPipelineDF]]
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
        shuffle_features: Optional[bool] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param grid: :class:`~gamma.ml.LearnerGrid` to be ranked \
            (either a single grid, or an iterable of multiple grids)
        :param cv: a cross validator (e.g., \
            :class:`~gamma.ml.validation.BootstrapCV`)
        :param scoring: a scorer to use when doing CV within GridSearch, defaults to \
            ``None``
        :param ranking_scorer: scoring function used for ranking across crossfit, \
            taking mean and standard deviation of the ranking scores_for_split and \
            returning the overall ranking score \
            (default: :meth:`.default_ranking_scorer`)
        :param shuffle_features: if ``True``, shuffle column order of features for \
            every crossfit (default: ``False``)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self._grids: Tuple[LearnerGrid, ...] = to_tuple(
            grid, element_type=LearnerGrid, arg_name="grid"
        )
        self._cv = cv
        self._scoring = scoring
        self._ranking_scorer = (
            LearnerRanker.default_ranking_scorer
            if ranking_scorer is None
            else ranking_scorer
        )
        self._shuffle_features = shuffle_features
        self._random_state = random_state

        # initialise state
        self._fit_params: Optional[Dict[str, Any]] = None
        self._ranking: Optional[List[LearnerScores]] = None
        self._best_model: Optional[T_LearnerPipelineDF] = None

    # add parameter documentation of ParallelizableMixin
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @staticmethod
    def default_ranking_scorer(scores: CrossfitScores) -> float:
        """
        The default function to determine the pipeline's rank: ``mean - 2 * std``.

        Its output is used to rank different parametrizations of one or more learners.

        :param scores: the validation scores for all crossfits
        :return: scalar score to be used for ranking the pipeline
        """
        return scores.mean() - 2 * scores.std()

    def fit(self: T_Self, sample: Sample, **fit_params) -> T_Self:
        """
        Rank the candidate learners and their hyper-parameter combinations using the
        given sample.

        :param sample: sample with which to fit the candidate learners from the grid(s)
        :param fit_params: any fit parameters to pass on to the learner's fit method
        """
        self: LearnerRanker[T_LearnerPipelineDF]  # support type hinting in PyCharm

        ranking: List[LearnerScores[T_LearnerPipelineDF]] = self._rank_learners(
            sample=sample, **fit_params
        )
        ranking.sort(key=lambda le: le.ranking_score, reverse=True)

        self._fit_params = fit_params
        self._ranking = ranking
        self._best_model = self._ranking[0].pipeline.fit(
            X=sample.features, y=sample.target
        )

        return self

    @property
    def is_fitted(self) -> bool:
        """``True`` if this ranker is fitted, ``False`` otherwise."""
        return self._ranking is not None

    def ranking(self) -> List[LearnerScores[T_LearnerPipelineDF]]:
        """
        :return a ranking of all learners that were evaluated based on the parameter
        grids passed to this ranker, in descending order of the ranking score.
        """
        self._ensure_fitted()
        return self._ranking.copy()

    @property
    def best_model(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample
        """
        self._ensure_fitted()
        return self._best_model

    @property
    def best_model_crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        The crossfit for the best model, fitted with the same sample and fit
        parameters used to fit this ranker.
        """
        self._ensure_fitted()
        return self._best_crossfit

    def summary_report(self, max_learners: Optional[int] = None) -> str:
        """
        Return a human-readable report of learner validation results, sorted by
        ranking score in descending order.

        :param max_learners: maximum number of learners to include in the report \
            (optional)

        :return: a summary string of the pipeline ranking
        """

        self._ensure_fitted()

        def _model_name(evaluation: LearnerScores) -> str:
            return type(evaluation.pipeline.final_estimator).__name__

        def _parameters(params: Mapping[str, Iterable[Any]]) -> str:
            return ",".join(
                [
                    f"{param_name}={param_value}"
                    for param_name, param_value in params.items()
                ]
            )

        ranking = self._ranking[:max_learners] if max_learners else self._ranking

        name_width = max([len(_model_name(ranked_model)) for ranked_model in ranking])

        return "\n".join(
            [
                f"Rank {rank + 1:2d}: "
                f"{_model_name(evaluation):>{name_width}s}, "
                f"ranking_score={evaluation.ranking_score:9.3g}, "
                f"scores_mean={evaluation.scores.mean():9.3g}, "
                f"scores_std={evaluation.scores.std():9.3g}, "
                f"parameters={{{_parameters(evaluation.parameters)}}}"
                "\n"
                for rank, evaluation in enumerate(ranking)
            ]
        )

    def _rank_learners(
        self, sample: Sample, **fit_params
    ) -> List[LearnerScores[T_LearnerPipelineDF]]:
        ranking_scorer = self._ranking_scorer

        configurations = (
            (grid.pipeline.clone().set_params(**parameters), parameters)
            for grid in self._grids
            for parameters in grid
        )

        ranking: List[LearnerScores[T_LearnerPipelineDF]] = []
        best_score: float = -math.inf
        best_crossfit: Optional[LearnerCrossfit[T_LearnerPipelineDF]] = None

        for pipeline, parameters in configurations:
            crossfit = LearnerCrossfit(
                pipeline=pipeline,
                cv=self._cv,
                shuffle_features=self._shuffle_features,
                random_state=self._random_state,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )

            pipeline_scoring: CrossfitScores = crossfit.fit_score(
                sample=sample, scoring=self._scoring, **fit_params
            )

            ranking_score = ranking_scorer(pipeline_scoring)

            ranking.append(
                LearnerScores(
                    pipeline=pipeline,
                    parameters=parameters,
                    scores=pipeline_scoring,
                    ranking_score=ranking_score,
                )
            )

            if ranking_score > best_score:
                best_score = ranking_score
                best_crossfit = crossfit

        self._best_crossfit = best_crossfit
        return ranking
