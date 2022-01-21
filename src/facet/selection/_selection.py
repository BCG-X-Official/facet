"""
Core implementation of :mod:`facet.selection`
"""
import inspect
import itertools
import logging
import operator
import re
from functools import reduce
from itertools import chain
from re import Pattern
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
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from pytools.api import AllTracker, deprecated, inheritdoc
from pytools.fit import FittableMixin
from pytools.parallelization import ParallelizableMixin
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF

from facet.data import Sample
from facet.selection.base import BaseParameterSpace

log = logging.getLogger(__name__)

__all__ = ["LearnerGrid", "LearnerEvaluation", "LearnerRanker"]

#
# Type constants
#

# sklearn does not publish base class BaseSearchCV, so we pull it from the MRO
# of GridSearchCV
BaseSearchCV = [
    base_class
    for base_class in GridSearchCV.mro()
    if base_class.__name__ == "BaseSearchCV"
][0]

#
# Type variables
#

T_Self = TypeVar("T_Self")
T_LearnerPipelineDF = TypeVar(
    "T_LearnerPipelineDF", RegressorPipelineDF, ClassifierPipelineDF
)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)
T_SearchCV = TypeVar("T_SearchCV", bound=BaseSearchCV)

#
# Constants
#

ARG_SAMPLE_WEIGHT = "sample_weight"

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="[see superclass]")
class LearnerRanker(
    FittableMixin[Sample], ParallelizableMixin, Generic[T_LearnerPipelineDF, T_SearchCV]
):
    """
    Score and rank different parametrizations of one or more learners,
    using cross-validation.

    The learner ranker can run a simultaneous grid search across multiple alternative
    learner pipelines, supporting the ability to simultaneously select a learner
    algorithm and optimize hyper-parameters.
    """

    #: The searcher used to fit this LearnerRanker; ``None`` if not fitted.
    searcher_: Optional[T_SearchCV]

    _CV_RESULT_COLUMNS = [
        r"mean_test_\w+",
        r"std_test_\w+",
        r"param_\w+",
        r"(rank|mean|std)_\w+",
    ]

    # noinspection PyTypeChecker
    _CV_RESULT_PATTERNS: List[Pattern] = list(map(re.compile, _CV_RESULT_COLUMNS))
    _DEFAULT_REPORT_SORT_COLUMN = "rank_test_score"

    def __init__(
        self,
        searcher_factory: Callable[..., T_SearchCV],
        parameter_space: BaseParameterSpace,
        *,
        cv: Optional[BaseCrossValidator] = None,
        scoring: Union[str, Callable[[float, float], float], None] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
        **searcher_params: Any,
    ) -> None:
        """
        :param searcher_factory: a cross-validation searcher class, or any other
            callable that instantiates a cross-validation searcher
        :param parameter_space: the parameter space to search
        :param cv: a cross validator (e.g.,
            :class:`.BootstrapCV`)
        :param scoring: a scoring function (by name, or as a callable) for evaluating
            learners (optional; use learner's default scorer if not specified here).
            If passing a callable, the ``"score"`` will be used as the name of the
            scoring function unless the callable defines a ``__name__`` attribute
        :param random_state: optional random seed or random state for shuffling the
            feature column order
        %%PARALLELIZABLE_PARAMS%%
        :param searcher_params: additional parameters to be passed on to the searcher;
            must not include the first two positional arguments of the searcher
            constructor used to pass the estimator and the search space, since these
            will be populated using arg parameter_space
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self.searcher_factory = searcher_factory
        self.parameter_space = parameter_space
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.searcher_params = searcher_params

        #
        # validate parameters for the searcher factory
        #

        searcher_factory_params = inspect.signature(searcher_factory).parameters.keys()

        # raise an error if the searcher params include the searcher's first two
        # positional arguments
        reserved_params = set(itertools.islice(searcher_factory_params, 2))

        reserved_params_overrides = reserved_params.intersection(searcher_params.keys())

        if reserved_params_overrides:
            raise ValueError(
                "arg searcher_params must not include the first two positional "
                "arguments of arg searcher_factory, but included: "
                + ", ".join(reserved_params_overrides)
            )

        # raise an error if the searcher does not support any of the given parameters
        unsupported_params = set(self._get_searcher_parameters().keys()).difference(
            searcher_factory_params
        )

        if unsupported_params:
            raise TypeError(
                "parameters not supported by arg searcher_factory: "
                + ", ".join(unsupported_params)
            )

        self.searcher_ = None

    __init__.__doc__ = __init__.__doc__.replace(
        "%%PARALLELIZABLE_PARAMS%%", ParallelizableMixin.__init__.__doc__.strip()
    )

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.searcher_ is not None

    @property
    def best_estimator_(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample.
        """
        self._ensure_fitted()
        searcher = self.searcher_
        if searcher.refit:
            return searcher.best_estimator_
        else:
            raise AttributeError(
                "best_model_ is not defined; use a CV searcher with refit=True"
            )

    def fit(
        self: T_Self,
        sample: Sample,
        groups: Union[pd.Series, np.ndarray, Sequence, None] = None,
        **fit_params: Any,
    ) -> T_Self:
        """
        Rank the candidate learners and their hyper-parameter combinations using
        crossfits from the given sample.

        Other than the scikit-learn implementation of grid search, arbitrary parameters
        can be passed on to the learner pipeline(s) to be fitted.

        :param sample: the sample from which to fit the crossfits
        :param groups:
        :param fit_params: any fit parameters to pass on to the learner's fit method
        :return: ``self``
        """
        self: LearnerRanker[
            T_LearnerPipelineDF, T_SearchCV
        ]  # support type hinting in PyCharm

        self._reset_fit()

        if ARG_SAMPLE_WEIGHT in fit_params:
            raise ValueError(
                "arg sample_weight is not supported, use ag sample.weight instead"
            )

        if isinstance(groups, pd.Series):
            if not groups.index.equals(sample.index):
                raise ValueError(
                    "index of arg groups is not equal to index of arg sample"
                )
        elif groups is not None:
            if len(groups) != len(sample):
                raise ValueError(
                    "length of arg groups is not equal to length of arg sample"
                )

        parameter_space = self.parameter_space
        searcher: BaseSearchCV
        searcher = self.searcher_ = self.searcher_factory(
            parameter_space.estimator,
            parameter_space.parameters,
            **self._get_searcher_parameters(),
        )
        if sample.weight is not None:
            fit_params = {ARG_SAMPLE_WEIGHT: sample.weight, **fit_params}

        searcher.fit(X=sample.features, y=sample.target, groups=groups, **fit_params)

        return self

    def summary_report(self, *, sort_by: Optional[str] = None) -> pd.DataFrame:
        """
        Create a summary table of the scores achieved by all learners in the grid
        search, sorted by ranking score in descending order.

        :param sort_by: name of the column to sort the report by, in ascending order,
            if the column is present (default: ``"%%SORT_COLUMN%%"``)

        :return: the summary report of the grid search as a data frame
        """

        self._ensure_fitted()

        if sort_by is None:
            sort_by = self._DEFAULT_REPORT_SORT_COLUMN

        cv_results: Dict[str, Any] = self.searcher_.cv_results_

        # we create a table using a subset of the cv results, to keep the report
        # relevant and readable
        cv_results_subset: Dict[str, np.ndarray] = {}

        # add the sorting column as the leftmost column of the report
        sort_results = sort_by in cv_results
        if sort_results:
            cv_results_subset[sort_by] = cv_results[sort_by]

        # add all other columns that match any of the pre-defined patterns
        for pattern in self._CV_RESULT_PATTERNS:
            cv_results_subset.update(
                {
                    name: values
                    for name, values in cv_results.items()
                    if name not in cv_results_subset and pattern.fullmatch(name)
                }
            )

        # convert the results into a data frame and sort
        report = pd.DataFrame(cv_results_subset)

        # split column headers containing one or more "__",
        # resulting in a column MultiIndex

        report.columns = report.columns.str.split("__", expand=True).map(
            lambda column: tuple(level if pd.notna(level) else "" for level in column)
        )

        # sort the report, if applicable
        if sort_results:
            report = report.sort_values(by=sort_by)

        return report

    def _reset_fit(self) -> None:
        # make this object not fitted
        self.searcher_ = None

    def _get_searcher_parameters(self) -> Dict[str, Any]:
        # make a dict of all parameters to be passed to the searcher
        return {
            **{
                k: v
                for k, v in dict(
                    cv=self.cv,
                    scoring=self.scoring,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                    shared_memory=self.shared_memory,
                    pre_dispatch=self.pre_dispatch,
                    verbose=self.verbose,
                ).items()
                if v is not None
            },
            **self.searcher_params,
        }

    summary_report.__doc__ = summary_report.__doc__.replace(
        "%%SORT_COLUMN%%", _DEFAULT_REPORT_SORT_COLUMN
    )


class LearnerGrid(Generic[T_LearnerPipelineDF]):
    """
    A grid of hyper-parameters for tuning a learner pipeline.
    """

    @deprecated(message=f"use class {LearnerRanker.__name__} instead")
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

    @deprecated(message=f"use class {LearnerRanker.__name__} instead")
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


__tracker.validate()
