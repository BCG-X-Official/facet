"""
Core implementation of :mod:`facet.selection`
"""

import logging
import math
import operator
from functools import reduce
from itertools import chain
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator

from pytools.api import AllTracker, inheritdoc, to_tuple
from pytools.fit import FittableMixin
from pytools.parallelization import JobRunner, ParallelizableMixin
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from facet.crossfit import LearnerCrossfit
from facet.data import Sample

log = logging.getLogger(__name__)

__all__ = ["LearnerGrid", "LearnerEvaluation", "LearnerRanker"]

#
# Type variables
#

T_Self = TypeVar("T_Self")
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


class LearnerGrid(Generic[T_LearnerPipelineDF]):
    """
    A grid of hyper-parameters for tuning a learner pipeline.
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        learner_parameters: Dict[str, Sequence],
        preprocessing_parameters: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        """
        :param pipeline: the :class:`~.sklearndf.pipeline.RegressorPipelineDF` or
            :class:`~.sklearndf.pipeline.ClassifierPipelineDF` to which the
            hyper-parameters will be applied
        :param learner_parameters: the hyper-parameter grid in which to search for the
            optimal parameter values for the pipeline's final estimator
        :param preprocessing_parameters: the hyper-parameter grid in which to search
            for the optimal parameter values for the pipeline's preprocessing pipeline
            (optional)
        """
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
        """
        The parameter grid for the entire pipeline.
        """
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


class LearnerEvaluation(Generic[T_LearnerPipelineDF]):
    """
    A collection of scores for a specific parametrization of a learner pipeline,
    generated by a :class:`.LearnerRanker`.
    """

    __slots__ = ["pipeline", "parameters", "scoring_name", "scores", "ranking_score"]

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scoring_name: str,
        scores: np.ndarray,
        ranking_score: float,
    ) -> None:
        """
        :param pipeline: the unfitted learner pipeline
        :param parameters: the hyper-parameters for which the learner pipeline was
            scored, as a mapping of parameter names to parameter values
        :param scoring_name: the name of the scoring function used to calculate the
            scores
        :param scores: the scores of all crossfits of the learner pipeline
        :param ranking_score: the aggregate score determined by the ranking
            metric of :class:`.LearnerRanker`, used for ranking the learners
        """
        super().__init__()

        #: The unfitted learner pipeline.
        self.pipeline = pipeline

        #: The hyper-parameters for which the learner pipeline was scored.
        self.parameters = parameters

        #: The name of the scoring function used to calculate the scores.
        self.scoring_name = scoring_name

        #: The scores of all crossfits of the learner pipeline.
        self.scores = scores

        #: The aggregate score determined by the ranking metric of
        #: :class:`.LearnerRanker`, used for ranking the learners.
        self.ranking_score = ranking_score


@inheritdoc(match="[see superclass]")
class LearnerRanker(
    ParallelizableMixin, FittableMixin[Sample], Generic[T_LearnerPipelineDF]
):
    """
    Score and rank different parametrizations of one or more learners,
    using cross-validation.

    The learner ranker can run a simultaneous grid search across multiple alternative
    learner pipelines, supporting the ability to simultaneously select a learner
    algorithm and optimize hyper-parameters.
    """

    def __init__(
        self,
        grids: Union[
            LearnerGrid[T_LearnerPipelineDF], Iterable[LearnerGrid[T_LearnerPipelineDF]]
        ],
        cv: Optional[BaseCrossValidator],
        scoring: Union[str, Callable[[float, float], float], None] = None,
        ranking_scorer: Callable[[np.ndarray], float] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param grids: learner grids to be ranked
            (either a single grid, or an iterable of multiple grids)
        :param cv: a cross validator (e.g.,
            :class:`.BootstrapCV`)
        :param scoring: a scoring function (by name, or as a callable) for evaluating
            learners (optional; use learner's default scorer if not specified here).
            If passing a callable, the ``"score"`` will be used as the name of the
            scoring function unless the callable defines a ``__name__`` attribute
        :param ranking_scorer: a function to calculate a scalar score for every
            crossfit and returning a float.
            The resulting score is used to rank all crossfits (highest score is best).
            Defaults to :meth:`.default_ranking_scorer`, calculating
            `mean(scores) - 2 * std(scores, ddof=1)`
        :param random_state: optional random seed or random state for shuffling the
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

        grids_tuple: Tuple[LearnerGrid, ...] = to_tuple(
            grids, element_type=LearnerGrid, arg_name="grids"
        )
        if len(grids_tuple) == 0:
            raise ValueError("arg grids must specify at least one LearnerGrid")
        learner_type = _learner_type(grids_tuple[0].pipeline)
        if not all(isinstance(grid.pipeline, learner_type) for grid in grids_tuple[1:]):
            raise ValueError("arg grids mixes regressor and classifier pipelines")

        self.grids = grids_tuple
        self.cv = cv
        self.scoring = scoring
        self.ranking_scorer = (
            LearnerRanker.default_ranking_scorer
            if ranking_scorer is None
            else ranking_scorer
        )
        self.random_state = random_state

        # initialise state
        self._ranking: Optional[List[LearnerEvaluation]] = None
        self._best_model: Optional[T_LearnerPipelineDF] = None

    # add parameter documentation of ParallelizableMixin
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self._ranking is not None

    @property
    def scoring_name(self) -> str:
        """
        The name of the scoring function used to rank the learners.
        """
        scoring = self.scoring
        if isinstance(scoring, str):
            return scoring
        elif callable(scoring):
            try:
                return scoring.__name__
            except AttributeError:
                return "score"
        else:
            learner_type = _learner_type(self.grids[0].pipeline)
            if learner_type is RegressorPipelineDF:
                return "r2_score"
            elif learner_type is ClassifierPipelineDF:
                return "accuracy_score"
            else:
                # default case - we should not end up here but adding this for forward
                # compatibility
                return "score"

    @property
    def ranking_(self) -> List[LearnerEvaluation[T_LearnerPipelineDF]]:
        """
        A list of :class:`.LearnerEvaluation` for all learners evaluated
        by this ranker, in descending order of the ranking score.
        """
        self._ensure_fitted()
        return self._ranking

    @property
    def best_model_(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample.
        """
        self._ensure_fitted()
        return self._best_model

    @property
    def best_model_crossfit_(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        The crossfit which obtained the best ranking score.
        """
        self._ensure_fitted()
        return self._best_crossfit

    @staticmethod
    def default_ranking_scorer(scores: np.ndarray) -> float:
        """
        The default function used to rank pipelines.

        Calculates `mean(scores) - 2 * std(scores, ddof=1)`, i.e., ranks pipelines by a
        (pessimistic) lower bound of the expected score.

        :param scores: the scores for all crossfits
        :return: scalar score for ranking the pipeline
        """
        return scores.mean() - 2 * scores.std(ddof=1)

    def fit(self: T_Self, sample: Sample, **fit_params: Any) -> T_Self:
        """
        Rank the candidate learners and their hyper-parameter combinations using
        crossfits from the given sample.

        Other than the scikit-learn implementation of grid search, arbitrary parameters
        can be passed on to the learner pipeline(s) to be fitted.

        :param sample: the sample from which to fit the crossfits
        :param fit_params: any fit parameters to pass on to the learner's fit method
        :return: ``self``
        """
        self: LearnerRanker[T_LearnerPipelineDF]  # support type hinting in PyCharm

        ranking: List[LearnerEvaluation[T_LearnerPipelineDF]] = self._rank_learners(
            sample=sample, **fit_params
        )
        ranking.sort(key=lambda le: le.ranking_score, reverse=True)

        self._ranking = ranking
        self._best_model = self._ranking[0].pipeline.fit(
            X=sample.features, y=sample.target
        )

        return self

    def summary_report(self) -> pd.DataFrame:
        """
        Create a summary table of the scores achieved by all learners in the grid
        search, sorted by ranking score in descending order.

        :return: the summary report of the grid search as a data frame
        """

        self._ensure_fitted()

        # define the columns of the resulting data frame

        col_ranking_score = "ranking_score"
        scoring_name = self.scoring_name
        col_scores_mean = f"{scoring_name}__mean"
        col_scores_std = f"{scoring_name}__std"
        col_learner_type = f"{self.grids[0].pipeline.final_estimator_name}__type"

        parameters: List[str] = []
        for grid in self.grids:
            # noinspection PyTypeChecker
            parameters.extend(grid.parameters.keys() - parameters)

        columns = [
            col_ranking_score,
            col_scores_mean,
            col_scores_std,
            col_learner_type,
            *parameters,
        ]

        # build the report

        report = pd.DataFrame.from_records(
            [
                {
                    col_ranking_score: evaluation.ranking_score,
                    col_scores_mean: evaluation.scores.mean(),
                    col_scores_std: evaluation.scores.std(ddof=1),
                    col_learner_type: type(
                        evaluation.pipeline.final_estimator
                    ).__name__,
                    **evaluation.parameters,
                }
                for evaluation in (
                    sorted(
                        self._ranking,
                        key=lambda evaluation: evaluation.ranking_score,
                        reverse=True,
                    )
                )
            ],
            columns=columns,
        ).rename_axis(index="rank")

        # split column headers containing one or more "__",
        # resulting in a column MultiIndex

        report.columns = report.columns.str.split("__", expand=True).map(
            lambda column: tuple(level if pd.notna(level) else "" for level in column)
        )

        return report

    def _rank_learners(
        self, sample: Sample, **fit_params
    ) -> List[LearnerEvaluation[T_LearnerPipelineDF]]:
        ranking_scorer = self.ranking_scorer

        pipelines: Iterable[T_LearnerPipelineDF]
        pipelines_parameters: Iterable[Dict[str, Any]]
        pipelines, pipelines_parameters = zip(
            *(
                (
                    cast(T_LearnerPipelineDF, grid.pipeline.clone()).set_params(
                        **parameters
                    ),
                    parameters,
                )
                for grid in self.grids
                for parameters in grid
            )
        )

        ranking: List[LearnerEvaluation[T_LearnerPipelineDF]] = []
        best_score: float = -math.inf
        best_crossfit: Optional[LearnerCrossfit[T_LearnerPipelineDF]] = None

        scoring_name = self.scoring_name

        crossfits = [
            LearnerCrossfit(
                pipeline=pipeline,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )
            for pipeline in pipelines
        ]

        queues = (
            crossfit.fit_score_queue(sample=sample, scoring=self.scoring, **fit_params)
            for crossfit in crossfits
        )

        pipeline_scorings: List[np.ndarray] = list(
            JobRunner.from_parallelizable(self).run_queues(*queues)
        )

        for crossfit, pipeline_parameters, pipeline_scoring in zip(
            crossfits, pipelines_parameters, pipeline_scorings
        ):

            ranking_score = ranking_scorer(pipeline_scoring)
            crossfit_pipeline = crossfit.pipeline
            assert crossfit_pipeline.is_fitted
            ranking.append(
                LearnerEvaluation(
                    pipeline=crossfit_pipeline,
                    parameters=pipeline_parameters,
                    scoring_name=scoring_name,
                    scores=pipeline_scoring,
                    ranking_score=ranking_score,
                )
            )

            if ranking_score > best_score:
                best_score = ranking_score
                best_crossfit = crossfit

        self._best_crossfit = best_crossfit
        return ranking


def _learner_type(pipeline: T_LearnerPipelineDF) -> Type[T_LearnerPipelineDF]:
    # determine whether a learner pipeline fits a regressor or a classifier
    for learner_type in [RegressorPipelineDF, ClassifierPipelineDF]:
        if isinstance(pipeline, learner_type):
            return learner_type
    if isinstance(pipeline, LearnerPipelineDF):
        raise TypeError(f"unknown learner pipeline type: {type(learner_type).__name__}")
    else:
        raise TypeError(
            "attribute grid.pipeline is not a learner pipeline: "
            f"{type(learner_type).__name__}"
        )


__tracker.validate()
