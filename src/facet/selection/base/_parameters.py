"""
Core implementation of :mod:`facet.selection.base`
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Sequence, TypeVar, Union

import pandas as pd
from scipy import stats

from pytools.api import AllTracker, inheritdoc
from pytools.expression import HasExpressionRepr
from sklearndf import ClassifierDF, EstimatorDF, RegressorDF, TransformerDF

log = logging.getLogger(__name__)

__all__ = [
    "BaseParameterSpace",
    "CandidateEstimatorDF",
]


#
# Type constants
#

ParameterDict = Dict[str, Union[List[Any], stats.rv_continuous, stats.rv_discrete]]


#
# Type variables
#

T_CandidateEstimatorDF = TypeVar("T_CandidateEstimatorDF", bound="CandidateEstimatorDF")
T_Estimator = TypeVar("T_Estimator", bound=EstimatorDF)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class BaseParameterSpace(HasExpressionRepr, Generic[T_Estimator], metaclass=ABCMeta):
    """
    A collection of parameters spanning a parameter space for hyper-parameter
    optimization.
    """

    def __init__(self, estimator: T_Estimator) -> None:
        """
        :param estimator: the estimator for which to specify parameter choices or
            distributions

        """
        self._estimator = estimator

    @property
    def estimator(self) -> T_Estimator:
        """
        The estimator associated with this parameter space.
        """
        return self._estimator

    @property
    def parameters(self) -> Union[List[ParameterDict], ParameterDict]:
        """
        The parameter choices (as lists) or distributions (from :mod:`scipy.stats`)
        that constitute this parameter space.

        This is a shortcut for calling method :meth:`.get_parameters` with no
        arguments.
        """
        return self.get_parameters()

    @abstractmethod
    def get_parameters(
        self, prefix: Optional[str] = None
    ) -> Union[List[ParameterDict], ParameterDict]:
        """
        Generate a dictionary of parameter choices and distributions,
        or a list of such dictionaries, compatible with `scikit-learn`'s CV search API
        (e.g., :class:`~sklearn.model_selection.GridSearchCV` or
        :class:`~sklearn.model_selection.RandomizedSearchCV`).

        :param prefix: an optional prefix to prepend to all parameter names in the
            resulting dictionary, separated by two underscore characters (``__``) as
            per `scikit-learn`'s convention for hierarchical parameter names
        :return: a dictionary mapping parameter names to parameter
            choices (as lists) or distributions (from :mod:`scipy.stats`)
        """
        pass


@inheritdoc(match="""[see superclass]""")
class CandidateEstimatorDF(ClassifierDF, RegressorDF, TransformerDF):
    """
    A trivial wrapper for classifiers, regressors and transformers, acting
    like a pipeline with a single step.

    Used in conjunction with :class:`MultiEstimatorParameterSpace` to evaluate multiple
    competing models: the :attr:`.candidate` parameter determines the estimator to be
    used and is used to include multiple estimators as part of the parameter space
    that is searched during model tuning.
    """

    #: name of the `candidate` parameter
    PARAM_CANDIDATE = "candidate"

    #: name of the `candidate_name` parameter
    PARAM_CANDIDATE_NAME = "candidate_name"

    #: The currently selected estimator candidate.
    candidate: Optional[Union[ClassifierDF, RegressorDF, TransformerDF]]

    #: The name of the candidate, used for more readable summary reports
    #: of model tuning results.
    candidate_name: Optional[str]

    def __init__(
        self,
        candidate: Optional[Union[ClassifierDF, RegressorDF, TransformerDF]] = None,
        candidate_name: Optional[str] = None,
    ) -> None:
        """
        :param candidate: the current estimator candidate; usually not specified on
            class creation but set as a parameter during multi-estimator model selection
        :param candidate_name: a name for the estimator candidate; usually not specified
            on class creation but set as a parameter during multi-estimator model
            selection
        """
        super().__init__()

        self.candidate = candidate
        self.candidate_name = candidate_name

    def _get_candidate(self) -> Union[ClassifierDF, RegressorDF, TransformerDF]:
        # get the estimator candidate; raise an attribute error if it has not been set

        if self.candidate is None:
            raise AttributeError("no candidate has been assigned")
        else:
            return self.candidate

    @property
    def classes_(self) -> Sequence[Any]:
        """[see superclass]"""
        return self._get_candidate().classes_

    # noinspection PyPep8Naming
    def predict_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""
        return self._get_candidate().predict_proba(X, **predict_params)

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""
        return self._get_candidate().predict_log_proba(X, **predict_params)

    # noinspection PyPep8Naming
    def decision_function(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self._get_candidate().decision_function(X, **predict_params)

    # noinspection PyPep8Naming
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """[see superclass]"""
        return self._get_candidate().score(X, y, sample_weight)

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self._get_candidate().predict(X, **predict_params)

    # noinspection PyPep8Naming
    def fit(
        self: T_CandidateEstimatorDF,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_CandidateEstimatorDF:
        """[see superclass]"""
        self._get_candidate().fit(X, y, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.candidate is not None and self.candidate.is_fitted

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""
        return self._get_candidate().inverse_transform(X)

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""
        return self._get_candidate().transform(X)

    @property
    def _estimator_type(self) -> str:
        # noinspection PyProtectedMember
        return self.candidate._estimator_type  # type: ignore

    def _get_features_in(self) -> pd.Index:
        return self._get_candidate().feature_names_in_

    def _get_n_outputs(self) -> int:
        return self._get_candidate().n_outputs_

    def _get_features_original(self) -> pd.Series:
        return self._get_candidate().feature_names_original_


__tracker.validate()
