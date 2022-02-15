"""
Core implementation of :mod:`facet.selection.base`
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from scipy import stats

from pytools.api import AllTracker
from pytools.expression import HasExpressionRepr
from sklearndf import EstimatorDF

log = logging.getLogger(__name__)

__all__ = [
    "BaseParameterSpace",
]


#
# Type constants
#

ParameterDict = Dict[str, Union[List[Any], stats.rv_continuous, stats.rv_discrete]]


#
# Type variables
#

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

    #: The estimator associated with this parameter space.
    estimator: T_Estimator

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
        The parameter choices and distributions that constitute this parameter space.

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
        or a list of such dictionaries, compatible with `scikit-learn`'s
        :class:`~sklearn.model_selection.GridSearchCV` and
        :class:`~sklearn.model_selection.RandomizedSearchCV`.

        :param prefix: an optional prefix to prepend to all parameter names in the
            resulting dictionary, separated by two underscore characters (`__`) as
            per scikit-learn's convention for hierarchical parameter names
        :return: a dictionary mapping parameter names to parameter
            distributions
        """
        pass


__tracker.validate()
