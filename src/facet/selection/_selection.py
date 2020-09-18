"""
Core implementation of :mod:`facet.selection`
"""

import logging
import math
import operator
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import *

import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator

from facet import Sample
from facet.crossfit import LearnerCrossfit
from pytools.api import AllTracker, inheritdoc, to_tuple
from pytools.fit import FittableMixin
from pytools.parallelization import ParallelizableMixin
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF

log = logging.getLogger(__name__)

__all__ = ["LearnerGrid", "LearnerScores", "LearnerRanker"]

#
# Type variables
#

T = TypeVar("T")
T_LearnerPipelineDF = TypeVar(
    "T_LearnerPipelineDF", RegressorPipelineDF, ClassifierPipelineDF
)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


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
    """

    __slots__ = ["pipeline", "parameters", "scores", "ranking_score"]

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scores: np.ndarray,
        ranking_score: float,
    ) -> None:
        """
        :param pipeline: the unfitted learner pipeline
        :param parameters: the hyper-parameters for which the learner pipeline was \
            scored, as a mapping of parameter names to parameter values
        :param scores: the scores of all crossfits of the learner pipeline
        :param ranking_score: the aggregate score determined by the ranking \
            metric of the :class:`.LearnerRanker`, used for ranking the learners
        """
        super().__init__()

        #: the unfitted learner pipeline
        self.pipeline = pipeline

        #: the hyper-parameters for which the learner pipeline was scored
        self.parameters = parameters

        #: the scores of all crossfits of the learner pipeline
        self.scores = scores

        #: overall score determined by the :class:`.LearnerRanker`, used for ranking
        #: the learners
        self.ranking_score = ranking_score


@inheritdoc(match="[see superclass")
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
        grids: Union[
            LearnerGrid[T_LearnerPipelineDF], Iterable[LearnerGrid[T_LearnerPipelineDF]]
        ],
        cv: Optional[BaseCrossValidator],
        scoring: Union[str, Callable[[float, float], float], None] = None,
        ranking_scorer: Callable[[np.ndarray], float] = None,
        shuffle_features: Optional[bool] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param grids: learner grids to be ranked \
            (either a single grid, or an iterable of multiple grids)
        :param cv: a cross validator (e.g., \
            :class:`.BootstrapCV`)
        :param scoring: a scoring function (by name or a callable) for evaluating \
            learners (optional; use learner's default scorer if not specified here)
        :param ranking_scorer: a function to calculate a scalar score for every \
            crossfit, taking a :class:`.CrossfitScores` and returning a float. \
            The resulting score is used to rank all crossfits (highest score is best). \
            Defaults to :meth:`.default_ranking_scorer`, calculating \
            `mean(scores) - 2 * std(scores)`.
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

        if scoring is not None and not (isinstance(scoring, str) or callable(scoring)):
            raise TypeError(
                "only a single scoring function is currently supported, "
                f"but a {type(scoring).__name__} was given as arg scoring"
            )

        self.grids: Tuple[LearnerGrid, ...] = to_tuple(
            grids, element_type=LearnerGrid, arg_name="grids"
        )
        self.cv = cv
        self.scoring = scoring
        self.ranking_scorer = (
            LearnerRanker.default_ranking_scorer
            if ranking_scorer is None
            else ranking_scorer
        )
        self.shuffle_features = shuffle_features
        self.random_state = random_state

        # initialise state
        self._ranking: Optional[List[LearnerScores]] = None
        self._best_model: Optional[T_LearnerPipelineDF] = None

    # add parameter documentation of ParallelizableMixin
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @staticmethod
    def default_ranking_scorer(scores: np.ndarray) -> float:
        """
        The default function used to rank pipelines.

        Calculates `mean(scores) - 2 * std(scores)`, i.e., ranks pipelines by a
        (pessimistic) lower bound of the expected score.

        :param scores: the scores for all crossfits
        :return: scalar score for ranking the pipeline
        """
        return scores.mean() - 2 * scores.std()

    def fit(self: T, sample: Sample, **fit_params) -> T:
        """
        Rank the candidate learners and their hyper-parameter combinations using
        crossfits from the given sample.

        Other than the scikit-learn implementation of grid search, arbitrary parameters
        can be passed on to the learner pipeline(s) to be fitted

        :param sample: the sample from which to fit the crossfits
        :param fit_params: any fit parameters to pass on to the learner's fit method
        """
        self: LearnerRanker[T_LearnerPipelineDF]  # support type hinting in PyCharm

        ranking: List[LearnerScores[T_LearnerPipelineDF]] = self._rank_learners(
            sample=sample, **fit_params
        )
        ranking.sort(key=lambda le: le.ranking_score, reverse=True)

        self._ranking = ranking
        self._best_model = self._ranking[0].pipeline.fit(
            X=sample.features, y=sample.target
        )

        return self

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._ranking is not None

    @property
    def ranking(self) -> List[LearnerScores[T_LearnerPipelineDF]]:
        """
        A list of :class:`.LearnerScorings` for all learners evaluated by this ranker, \
            in descending order of the ranking score.
        """
        self._ensure_fitted()
        return self._ranking.copy()

    @property
    def best_model(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample.
        """
        self._ensure_fitted()
        return self._best_model

    @property
    def best_model_crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        The crossfit which obtained the best ranking score.
        """
        self._ensure_fitted()
        return self._best_crossfit

    def summary_report(self, max_learners: Optional[int] = None) -> str:
        """
        A human-readable report of the learner evaluations, sorted by ranking score in
        descending order.

        :param max_learners: maximum number of learners to include in the report \
            (optional)

        :return: a multi-line string with a summary of the pipeline ranking
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
        ranking_scorer = self.ranking_scorer

        configurations: Iterable[Tuple[T_LearnerPipelineDF, Dict[str, Any]]] = (
            (
                cast(T_LearnerPipelineDF, grid.pipeline.clone()).set_params(
                    **parameters
                ),
                parameters,
            )
            for grid in self.grids
            for parameters in grid
        )

        ranking: List[LearnerScores[T_LearnerPipelineDF]] = []
        best_score: float = -math.inf
        best_crossfit: Optional[LearnerCrossfit[T_LearnerPipelineDF]] = None

        for pipeline, parameters in configurations:
            crossfit = LearnerCrossfit(
                pipeline=pipeline,
                cv=self.cv,
                shuffle_features=self.shuffle_features,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )

            pipeline_scoring: np.ndarray = crossfit.fit_score(
                sample=sample, scoring=self.scoring, **fit_params
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


__tracker.validate()
