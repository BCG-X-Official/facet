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
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin
from pytools.parallelization import ParallelizableMixin
from sklearndf import EstimatorDF
from sklearndf.pipeline import LearnerPipelineDF

from facet.data import Sample
from facet.selection.base import BaseParameterSpace, CandidateEstimatorDF

log = logging.getLogger(__name__)

__all__ = ["ModelSelector"]

#
# Type constants
#

# sklearn does not publish base class BaseSearchCV, so we pull it from the MRO
# of GridSearchCV
BaseSearchCV = next(
    filter(lambda cls: cls.__name__ == "BaseSearchCV", GridSearchCV.mro())
)

#
# Type variables
#

T_ModelSelector = TypeVar("T_ModelSelector", bound="ModelSelector")
T_EstimatorDF = TypeVar("T_EstimatorDF", bound=EstimatorDF)
# mypy - disabling due to lack of support for dynamic types
T_SearchCV = TypeVar("T_SearchCV", bound=BaseSearchCV)  # type: ignore

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
class ModelSelector(
    FittableMixin[Sample], ParallelizableMixin, Generic[T_EstimatorDF, T_SearchCV]
):
    """
    Select the best model obtained by fitting an estimator using different
    choices of hyper-parameters from a :class:`.ParameterSpace`, or even
    simultaneously evaluating multiple competing estimators from a
    :class:`.MultiEstimatorParameterSpace`.
    """

    #: A cross-validation searcher class, or any other callable
    #: that instantiates a cross-validation searcher, wrapped in
    #: a tuple to avoid confusion with methods
    searcher_type: Tuple[Callable[..., T_SearchCV]]

    #: The parameter space to search.
    parameter_space: BaseParameterSpace

    #: The cross-validator to be used by the searcher.
    cv: Optional[BaseCrossValidator]

    #: The scoring function (by name, or as a callable) to be used by the searcher
    #: (optional; use learner's default scorer if not specified here)
    scoring: Union[
        str,
        Callable[
            [EstimatorDF, pd.Series, pd.Series],
            float,
        ],
        None,
    ]

    #: Additional parameters to be passed on to the searcher.
    searcher_params: Dict[str, Any]

    #: The searcher used to fit this ModelSelector; ``None`` if not fitted.
    searcher_: Optional[T_SearchCV]

    # regular expressions and replacement patterns for selecting and renaming
    # relevant columns from scikit-learn's cv_result_ table
    _CV_RESULT_COLUMNS = [
        (r"rank_test_(\w+)", r"\1__test__rank"),
        (r"(mean|std)_test_(\w+)", r"\2__test__\1"),
        (r"param_(\w+)", r"param__\1"),
        (r"(rank|mean|std)_(\w+)_time", r"time__\2__\1"),
        (r"(rank|mean|std)_(\w+)_(\w+)", r"\3__\2__\1"),
    ]
    # noinspection PyTypeChecker
    _CV_RESULT_PATTERNS: List[Tuple[Pattern, str]] = [
        (re.compile(pattern), repl) for pattern, repl in _CV_RESULT_COLUMNS
    ]

    _CV_RESULT_CANDIDATE_PATTERN, _CV_RESULT_CANDIDATE_REPL = (
        re.compile(r"^(?:(param__)candidate__|param__(candidate(?:_name)?)$)"),
        r"\1\2",
    )

    # Default column to sort by in the summary_report() method.
    # This has no influence on how the best model is selected.
    _DEFAULT_REPORT_SORT_COLUMN = "rank_test_score"

    def __init__(
        self,
        searcher_type: Callable[..., T_SearchCV],
        parameter_space: BaseParameterSpace,
        *,
        cv: Optional[BaseCrossValidator] = None,
        scoring: Union[
            str,
            Callable[
                [EstimatorDF, pd.Series, pd.Series],
                float,
            ],
            None,
        ] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
        **searcher_params: Any,
    ) -> None:
        """
        :param searcher_type: a cross-validation searcher class, or any other
            callable that instantiates a cross-validation searcher
        :param parameter_space: the parameter space to search
        :param cv: the cross-validator to be used by the searcher
            (e.g., :class:`~sklearn.model_selection.RepeatedKFold`)
        :param scoring: a scoring function (by name, or as a callable) to be used by the
            searcher (optional; use learner's default scorer if not specified here).
            If passing a callable, ``"score"`` will be used as the name of the
            scoring function unless the callable defines a ``__name__`` attribute
        %%PARALLELIZABLE_PARAMS%%
        :param searcher_params: additional parameters to be passed on to the searcher;
            must not include the first two positional arguments of the searcher
            constructor used to pass the estimator and the search space, since these
            will be populated from arg ``parameter_space``
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self.searcher_type = (searcher_type,)
        self.parameter_space = parameter_space
        self.cv = cv
        self.scoring = scoring
        self.searcher_params = searcher_params

        #
        # validate parameters for the searcher factory
        #

        if not callable(searcher_type):
            raise TypeError(
                "arg searcher_type expected to be a callable, "
                f"but is a {type(searcher_type).__name__}"
            )

        searcher_factory_params = inspect.signature(searcher_type).parameters.keys()

        # raise an error if the searcher params include the searcher's first two
        # positional arguments
        reserved_params = set(itertools.islice(searcher_factory_params, 2))

        reserved_params_overrides = reserved_params.intersection(searcher_params.keys())

        if reserved_params_overrides:
            raise ValueError(
                "arg searcher_params must not include the first two positional "
                "arguments of arg searcher_type, but included: "
                + ", ".join(reserved_params_overrides)
            )

        # raise an error if the searcher does not support any of the given parameters
        unsupported_params = set(self._get_searcher_parameters().keys()).difference(
            searcher_factory_params
        )

        if unsupported_params:
            raise TypeError(
                "parameters not supported by arg searcher_type: "
                + ", ".join(unsupported_params)
            )

        self.searcher_ = None

    # mypy - incorrect type inference for __doc__
    __init__.__doc__ = cast(str, __init__.__doc__).replace(
        "%%PARALLELIZABLE_PARAMS%%",
        cast(str, ParallelizableMixin.__init__.__doc__).strip(),
    )

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.searcher_ is not None

    @property
    def best_estimator_(self) -> T_EstimatorDF:
        """
        The model which obtained the best ranking score, fitted on the entire sample.
        """
        self.ensure_fitted()
        searcher = self.searcher_
        assert searcher is not None, "Ranker is fitted"

        if searcher.refit:
            best_estimator = searcher.best_estimator_
            while isinstance(best_estimator, CandidateEstimatorDF):
                # unpack the candidate estimator
                best_estimator = best_estimator.candidate
            return best_estimator

        else:
            raise AttributeError(
                "best_estimator_ is not defined; use a CV searcher with refit=True"
            )

    def fit(  # type: ignore[override]
        # todo: remove 'type: ignore' once mypy correctly infers return type
        self: T_ModelSelector,
        sample: Sample,
        groups: Union[pd.Series, np.ndarray, Sequence, None] = None,
        **fit_params: Any,
    ) -> T_ModelSelector:
        """
        Search this model selector's parameter space to identify the model with the
        best-performing hyper-parameter combination, using the given sample to fit and
        score the candidate estimators.

        :param sample: the sample used to fit and score the estimators
        :param groups: group labels for the samples used while splitting the dataset
            into train/test set; passed on to the ``fit`` method of the searcher
        :param fit_params: parameters to pass on to the estimator's fit method
        :return: ``self``
        """

        self._reset_fit()

        if ARG_SAMPLE_WEIGHT in fit_params:
            raise ValueError(
                "arg sample_weight is not supported, use arg sample.weight instead"
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
        (searcher_type,) = self.searcher_type
        searcher = self.searcher_ = searcher_type(
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

        self.ensure_fitted()

        if sort_by is None:
            sort_by = self._DEFAULT_REPORT_SORT_COLUMN

        assert self.searcher_ is not None, "Ranker is fitted"
        cv_results: Dict[str, Any] = self.searcher_.cv_results_

        # we create a table using a subset of the cv results, to keep the report
        # relevant and readable
        cv_results_processed: Dict[str, np.ndarray] = {}

        unpack_candidate: bool = isinstance(
            self.parameter_space.estimator, CandidateEstimatorDF
        )

        def _process(name: str) -> Optional[str]:
            # process the name of the original cv_results_ record
            # to achieve a better table format

            match = pattern.fullmatch(name)
            if match is None:
                # we could not match the name:
                # return None so we don't include it in the summary report
                return None

            name = match.expand(repl)
            if unpack_candidate:
                # remove the "candidate" layer in the parameter output if we're dealing
                # with a multi parameter space
                return ModelSelector._CV_RESULT_CANDIDATE_PATTERN.sub(
                    ModelSelector._CV_RESULT_CANDIDATE_REPL, name
                )
            else:
                return name

        # add all columns that match any of the pre-defined patterns
        for pattern, repl in self._CV_RESULT_PATTERNS:
            cv_results_processed.update(
                {
                    name: (name_processed, values)
                    for name, name_processed, values in (
                        # iterate matches between pattern and name
                        (name, _process(name), values)
                        for name, values in cv_results.items()
                        if name not in cv_results_processed
                    )
                    if name_processed is not None
                }
            )

        # add the sorting column as the leftmost column of the report
        sort_column_processed: Optional[str]

        sort_column_processed, _ = cv_results_processed.get(sort_by, None)
        if sort_column_processed is None:
            sort_column_values = cv_results.get(sort_by, None)
            if sort_column_values is None:
                sort_column_processed = None
            else:
                sort_column_processed = sort_by
                cv_results_processed[sort_by] = cv_results[sort_by]

        # convert the results into a data frame and sort
        report = pd.DataFrame(
            {
                name_processed: values
                for name_processed, values in cv_results_processed.values()
            }
        )

        # sort the report, if applicable
        if sort_column_processed is not None:
            report = report.sort_values(by=sort_column_processed)

        # split column headers containing one or more "__",
        # resulting in a column MultiIndex

        report.columns = report.columns.str.split("__", expand=True).map(
            lambda column: tuple(level if pd.notna(level) else "-" for level in column)
        )

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
                    scoring=self._get_scorer(),
                    n_jobs=self.n_jobs,
                    shared_memory=self.shared_memory,
                    pre_dispatch=self.pre_dispatch,
                    verbose=self.verbose,
                ).items()
                if v is not None
            },
            **self.searcher_params,
        }

    def _get_scorer(
        self,
    ) -> Optional[Callable[[EstimatorDF, pd.DataFrame, pd.Series], float]]:
        scoring = self.scoring

        if scoring is None:
            return None

        elif isinstance(scoring, str):
            scorer = get_scorer(scoring)

        # noinspection PyPep8Naming
        def _scorer_fn(estimator: EstimatorDF, X: pd.DataFrame, y: pd.Series) -> float:
            while isinstance(estimator, CandidateEstimatorDF):
                assert estimator.candidate is not None, "estimator candidate is set"
                estimator = estimator.candidate

            if isinstance(estimator, LearnerPipelineDF):
                if estimator.preprocessing:
                    X = estimator.preprocessing.transform(X=X)
                estimator = estimator.final_estimator

            return scorer(estimator.native_estimator, X, y)

        return _scorer_fn

    # mypy - incorrect type inference for __doc__
    summary_report.__doc__ = summary_report.__doc__.replace(  # type: ignore
        "%%SORT_COLUMN%%", _DEFAULT_REPORT_SORT_COLUMN
    )


__tracker.validate()
