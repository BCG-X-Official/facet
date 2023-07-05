"""
Core implementation of :mod:`facet.selection`
"""
from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from scipy import stats
from sklearn.base import BaseEstimator
from typing_extensions import TypeAlias

from pytools.api import AllTracker, inheritdoc, subsdoc, to_list, validate_element_types
from pytools.expression import Expression, make_expression
from pytools.expression.atomic import Id
from sklearndf import EstimatorDF
from sklearndf.pipeline import LearnerPipelineDF, PipelineDF

from .base import BaseParameterSpace, CandidateEstimatorDF

log = logging.getLogger(__name__)

__all__ = [
    "MultiEstimatorParameterSpace",
    "ParameterSpace",
]


#
# Type aliases
#

ParameterSet: TypeAlias = Union[List[Any], stats.rv_continuous, stats.rv_discrete]
ParameterDict: TypeAlias = Dict[str, ParameterSet]

try:
    rv_frozen = next(
        t for t in type(stats.uniform()).mro() if t.__name__ == "rv_frozen"
    )
except StopIteration:
    warnings.warn(
        "stats.uniform() should be based on class rv_frozen", category=UserWarning
    )
    rv_frozen = type(None)


#
# Type variables
#

T_Estimator_co = TypeVar("T_Estimator_co", covariant=True, bound=EstimatorDF)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class ParameterSpace(BaseParameterSpace[T_Estimator_co], Generic[T_Estimator_co]):
    """
    A set of parameter choices or distributions spanning a parameter space for
    optimizing the hyperparameters of a single estimator.

    Parameter spaces provide an easy approach to define and validate search spaces
    for hyperparameter tuning of ML pipelines using `scikit-learn`'s
    :class:`~sklearn.model_selection.GridSearchCV` and
    :class:`~sklearn.model_selection.RandomizedSearchCV`.

    Parameter choices (as lists) or distributions (from :mod:`scipy.stats`) can be
    set using attribute access, and will be validated for correct names and values.

    Example:

    .. code-block:: python

        ps = ParameterSpace(
            RegressorPipelineDF(
                regressor=RandomForestRegressorDF(random_state=42),
                preprocessing=simple_preprocessor,
            ),
            name="random forest",
        )
        ps.regressor.min_weight_fraction_leaf = scipy.stats.loguniform(0.01, 0.1)
        ps.regressor.max_depth = [3, 4, 5, 7, 10]

        cv = RandomizedSearchCV(
            estimator=ps.estimator,
            param_distributions=ps.parameters,
            # ...
        )

        # The following will raise an AttributeError for the unknown attribute xyz:
        ps.regressor.xyz = [3, 4, 5, 7, 10]

        # the following will raise a TypeError because we do not assign a list or \
distribution:
        ps.regressor.max_depth = 3

    """

    def __init__(self, estimator: T_Estimator_co, name: Optional[str] = None) -> None:
        """
        :param estimator: the estimator to which to apply the parameters
        :param name: a name for the estimator to be used in summary reports;
            defaults to the name of the estimator's class, or the name of the final
            estimator's class if arg ``estimator`` is a pipeline
        """

        super().__init__(estimator=estimator)

        params: Dict[str, Any] = {
            name: param
            for name, param in estimator.get_params(deep=True).items()
            if "__" not in name
        }

        self._children: Dict[str, ParameterSpace[BaseEstimator]] = {
            name: ParameterSpace(estimator=value)
            for name, value in params.items()
            if isinstance(value, BaseEstimator)
        }

        self._name = name
        self._values: ParameterDict = {}
        self._params: Set[str] = set(params.keys())

    def get_name(self) -> str:
        """
        Get the name for this parameter space's estimator.

        If no name was passed to the constructor, determine the default name
        recursively as follows:

        - for meta-estimators, this is the default name of the delegate estimator
        - for pipelines, this is the default name of the final estimator
        - for all other estimators, this is the name of the estimator's class

        :return: the name for this parameter space's estimator
        """

        if self._name is None:
            return get_default_estimator_name(self._estimator)
        else:
            return self._name

    @subsdoc(
        pattern="or a list of such dictionaries, ",
        replacement="",
    )
    @subsdoc(
        pattern="one or more dictionaries, each mapping",
        replacement="a dictionary mapping",
        using=BaseParameterSpace.get_parameters,
    )
    def get_parameters(self, prefix: Optional[str] = None) -> ParameterDict:
        """[see superclass]"""

        return {
            "__".join(name): values
            for (name, values) in self._iter_parameters(
                path_prefix=[] if prefix is None else [prefix]
            )
        }

    def _validate_parameter(self, name: str, value: ParameterSet) -> None:
        if name not in self._params:
            raise AttributeError(
                f"unknown parameter name for "
                f"{type(self.estimator).__name__}: {name}"
            )

        if not (
            isinstance(
                value,
                (list, stats.rv_discrete, stats.rv_continuous),
            )
            or callable(getattr(value, "rvs", None))
        ):
            raise TypeError(
                f"expected list or distribution for parameter {name} but got: "
                f"{value!r}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._validate_parameter(name, value)
            if name in self.__dict__:
                warnings.warn(
                    f"parameter {name!r} overrides {type(self).__name__}"
                    f"attribute of same name",
                    stacklevel=-1,
                )
            self._values[name] = value

    def __dir__(self) -> Iterable[str]:
        return {*super().__dir__(), *self._params}

    def __getattr__(self, key: str) -> Any:
        if not key.startswith("_"):
            result: Union[ParameterSpace[Any], ParameterSet, None]

            result = self._children.get(key, None)
            if result is not None:
                return result

            result = self._values.get(key, None)
            if result is not None:
                return result

        return super().__getattribute__(key)

    def __iter__(self) -> Iterator[Tuple[List[str], ParameterSet]]:
        return self._iter_parameters([])

    def _iter_parameters(
        self, path_prefix: List[str]
    ) -> Iterator[Tuple[List[str], ParameterSet]]:
        yield from (
            ([*path_prefix, name], value) for name, value in self._values.items()
        )

        for name, child in self._children.items():
            yield from child._iter_parameters([*path_prefix, name])

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return self._to_expression([])

    def _to_expression(self, path_prefix: Union[str, List[str]]) -> Expression:
        # path_prefix: the path prefix to prepend to each parameter name

        def _values_to_expression(values: ParameterSet) -> Expression:
            if isinstance(values, rv_frozen):
                # disabling type-checks: mypy cannot access the class signature
                # of private class rv_frozen, which is obtained dynamically at runtime
                return Id(values.dist.name)(*values.args, **values.kwds)  # type: ignore
            elif isinstance(values, (stats.rv_continuous, stats.rv_discrete)):
                try:
                    return Id(values.name)(values.a, values.b)
                except AttributeError:
                    pass

            return make_expression(values)

        path_prefix_list: List[str] = to_list(
            path_prefix, element_type=str, optional=True, arg_name="path_prefix"
        )

        parameters = {
            ".".join(path): _values_to_expression(value)
            for path, value in self._iter_parameters(path_prefix=path_prefix_list)
        }

        if path_prefix_list:
            return Id(type(self))(
                **{".".join(path_prefix_list): self.estimator}, **parameters
            )
        else:
            return Id(type(self))(self.estimator, **parameters)


@inheritdoc(match="""[see superclass]""")
class MultiEstimatorParameterSpace(
    BaseParameterSpace[T_Estimator_co], Generic[T_Estimator_co]
):
    """
    A collection of parameter spaces, each representing a competing estimator from which
    to select the best-performing candidate with optimal hyperparameters.

    See :class:`.ParameterSpace` for details on setting up and using parameter spaces.
    """

    #: The parameter spaces constituting this multi-estimator parameter space.
    spaces: Tuple[ParameterSpace[T_Estimator_co], ...]

    def __init__(self, *spaces: ParameterSpace[T_Estimator_co]) -> None:
        """
        :param spaces: the parameter spaces from which to select the best estimator
        """
        validate_element_types(spaces, expected_type=ParameterSpace, name="arg spaces")
        validate_spaces(spaces)

        if len(spaces) == 0:
            raise TypeError("no parameter space passed; need to pass at least one")

        super().__init__(estimator=CandidateEstimatorDF())

        self.spaces = spaces

    @subsdoc(
        pattern=(
            r"a dictionary of parameter choices and distributions,[\n\s]*"
            r"or a list of such dictionaries"
        ),
        replacement="a list of dictionaries of parameter distributions",
    )
    @subsdoc(
        pattern="one or more dictionaries,",
        replacement="a list of dictionaries,",
        using=BaseParameterSpace.get_parameters,
    )
    def get_parameters(self, prefix: Optional[str] = None) -> List[ParameterDict]:
        """[see superclass]"""
        if prefix is None:
            prefix = ""
            candidate_prefixed = CandidateEstimatorDF.PARAM_CANDIDATE
        else:
            prefix = f"{prefix}__"
            candidate_prefixed = prefix + CandidateEstimatorDF.PARAM_CANDIDATE

        return [
            {
                candidate_prefixed: [space.estimator],
                prefix + CandidateEstimatorDF.PARAM_CANDIDATE_NAME: [space.get_name()],
                **space.get_parameters(prefix=candidate_prefixed),
            }
            for space in self.spaces
        ]

    def to_expression(self) -> Expression:
        """[see superclass]"""
        # noinspection PyProtectedMember
        return Id(type(self))(*self.spaces)


__tracker.validate()


#
# auxiliary functions
#


def ensure_subclass(
    estimator_type: Type[T_Estimator_co], expected_type: Type[T_Estimator_co]
) -> None:
    """
    Ensure that the given estimator type is a subclass of the expected estimator type.

    :param estimator_type: the estimator type to validate
    :param expected_type: the expected estimator type
    """
    if not issubclass(estimator_type, expected_type):
        raise TypeError(
            f"arg estimator_type must be a subclass of {expected_type.__name__} "
            f"but is: {estimator_type.__name__}"
        )


def validate_spaces(spaces: Collection[ParameterSpace[T_Estimator_co]]) -> None:
    """
    Ensure that all parameter spaces use the same estimator type (typically regressors
    or classifiers)

    :param spaces: the parameter spaces to check
    """

    estimator_types: Set[str] = {
        getattr(space.estimator, "_estimator_type") for space in spaces
    }

    if len(estimator_types) > 1:
        raise TypeError(
            "all parameter spaces must use the same estimator type, "
            "but got multiple types: " + ", ".join(sorted(estimator_types))
        )


def get_default_estimator_name(estimator: EstimatorDF) -> str:
    """
    Get a default name of the estimator.

    - for meta-estimators, this is the default name of the delegate estimator
    - for pipelines, this is the default name of the final estimator
    - for all other estimators, this is the name of the estimator's class


    :param estimator: the estimator to get the default name for
    :return: the default name
    """

    while True:
        if isinstance(estimator, CandidateEstimatorDF):
            if estimator.candidate is None:
                return type(estimator).__name__
            else:
                estimator = estimator.candidate

        elif isinstance(estimator, PipelineDF) and estimator.steps:
            estimator = estimator.steps[-1]

        elif isinstance(estimator, LearnerPipelineDF):
            estimator = estimator.final_estimator

        else:
            return type(estimator).__name__
