"""
Core implementation of :mod:`facet.selection`
"""
import inspect
import itertools
import logging
import re
from re import Pattern
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin
from pytools.parallelization import ParallelizableMixin
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from facet.data import Sample
from facet.selection.base import BaseParameterSpace

log = logging.getLogger(__name__)

__all__ = ["LearnerRanker"]

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

        if type(self.scoring) == str:
            self.scoring = self._preprocess_scoring(self.scoring)

        self.searcher_ = None

    __init__.__doc__ = __init__.__doc__.replace(
        "%%PARALLELIZABLE_PARAMS%%", ParallelizableMixin.__init__.__doc__.strip()
    )

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.searcher_ is not None

    @staticmethod
    def _preprocess_scoring(scoring: str):
        def _score_fn(estimator, X: pd.DataFrame, y: pd.Series):
            estimator = estimator.candidate

            if isinstance(estimator, LearnerPipelineDF):
                if estimator.preprocessing:
                    X = estimator.preprocessing.transform(X=X)
                estimator = estimator.final_estimator

            scorer = check_scoring(
                estimator=estimator.native_estimator,
                scoring=scoring,
            )

            return scorer(estimator.native_estimator, X.values, y.values)

        return _score_fn

    @property
    def best_estimator_(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample.
        """
        self._ensure_fitted()
        searcher = self.searcher_
        if searcher.refit:
            return searcher.best_estimator_.candidate
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


__tracker.validate()
