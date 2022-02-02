"""
Core implementation of :mod:`facet.selection`
"""

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
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator

from pytools.api import AllTracker, inheritdoc, subsdoc, to_list, validate_element_types
from pytools.expression import Expression, make_expression
from pytools.expression.atomic import Id
from sklearndf import ClassifierDF, EstimatorDF, RegressorDF, TransformerDF

from .base import BaseParameterSpace

log = logging.getLogger(__name__)

__all__ = [
    "CandidateEstimatorDF",
    "MultiEstimatorParameterSpace",
    "ParameterSpace",
]


#
# Type constants
#

ParameterSet = Union[List[Any], stats.rv_continuous, stats.rv_discrete]
ParameterDict = Dict[str, ParameterSet]

rv_frozen = type(stats.uniform())
assert rv_frozen.__name__ == "rv_frozen"


#
# Type variables
#

T_Self = TypeVar("T_Self")
T_Candidate_co = TypeVar("T_Candidate_co", covariant=True, bound=EstimatorDF)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class ParameterSpace(BaseParameterSpace[T_Candidate_co], Generic[T_Candidate_co]):
    """
    A set of parameters spanning a parameter space for optimizing the hyper-parameters
    of a single estimator.

    Parameter spaces provide an easy approach to define and validate search spaces
    for hyper-parameter tuning of ML pipelines using `scikit-learn`'s
    :class:`~sklearn.model_selection.GridSearchCV` and
    :class:`~sklearn.model_selection.RandomizedSearchCV`.

    Parameter lists or distributions to be searched can be set using attribute access,
    and will be validated for correct names and values.

    Example:

    .. code-block:: python

        ps = ParameterSpace(
            RegressorPipelineDF(
                regressor=RandomForestRegressorDF(random_state=42),
                preprocessing=simple_preprocessor,
            ),
            candidate_name="rf_candidate"
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

    def __init__(
        self, candidate: T_Candidate_co, candidate_name: Optional[str] = None
    ) -> None:
        """
        :param candidate: the estimator candidate to which to apply the parameters to
        :param candidate_name: a name for the estimator candidate to be used in summary
            reports
        """

        super().__init__(estimator=CandidateEstimatorDF(candidate, candidate_name))

        params: Dict[str, Any] = {
            name: param
            for name, param in candidate.get_params(deep=True).items()
            if "__" not in name
        }

        self._children: Dict[str, ParameterSpace] = {
            name: ParameterSpace(candidate=value)
            for name, value in params.items()
            if isinstance(value, BaseEstimator)
        }
        self._values: ParameterDict = {}
        self._params: Set[str] = set(params.keys())

    @subsdoc(
        pattern="or a list of such dictionaries, ",
        replacement="",
        using=BaseParameterSpace.get_parameters,
    )
    def get_parameters(self, prefix: Optional[str] = None) -> ParameterDict:
        """[see superclass]"""

        return {
            "__".join(name): values
            for (name, values) in self._iter_parameters(
                path_prefix=[
                    CandidateEstimatorDF.PARAM_CANDIDATE if prefix is None else prefix
                ]
            )
        }

    def _validate_parameter(self, name: str, value: ParameterSet) -> None:

        if name not in self._params:
            raise AttributeError(
                f"unknown parameter name for "
                f"{type(self.estimator.candidate).__name__}: {name}"
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
                values: rv_frozen
                return Id(values.dist.name)(*values.args, **values.kwds)
            elif isinstance(values, (stats.rv_continuous, stats.rv_discrete)):
                try:
                    return Id(values.name)(values.a, values.b)
                except AttributeError:
                    pass

            return make_expression(values)

        path_prefix = (
            []
            if path_prefix is None
            else to_list(path_prefix, element_type=str, arg_name="path_prefix")
        )

        parameters = {
            ".".join(path): _values_to_expression(value)
            for path, value in self._iter_parameters(path_prefix=path_prefix)
        }

        if path_prefix:
            return Id(type(self))(
                **{".".join(path_prefix): self.estimator.candidate}, **parameters
            )
        else:
            return Id(type(self))(self.estimator.candidate, **parameters)


@inheritdoc(match="""[see superclass]""")
class MultiEstimatorParameterSpace(
    BaseParameterSpace[T_Candidate_co], Generic[T_Candidate_co]
):
    """
    A collection of parameter spaces, each representing a competing estimator from which
    select the best-performing candidate with optimal hyper-parameters.

    See :class:`.ParameterSpace` for documentation on how to set up and use parameter
    spaces.
    """

    def __init__(self, *spaces: ParameterSpace[T_Candidate_co]) -> None:
        """
        :param spaces: the parameter spaces from which to select the best estimator
        """
        validate_element_types(spaces, expected_type=ParameterSpace)
        validate_spaces(spaces)

        if len(spaces) == 0:
            raise TypeError("no parameter space passed; need to pass at least one")

        super().__init__(estimator=CandidateEstimatorDF.empty())

        self.spaces = spaces

    @subsdoc(
        pattern=(
            r"a dictionary of parameter distributions,[\n\s]*"
            r"or a list of such dictionaries"
        ),
        replacement="a list of dictionaries of parameter distributions",
        using=BaseParameterSpace.get_parameters,
    )
    def get_parameters(self, prefix: Optional[str] = None) -> List[ParameterDict]:
        """[see superclass]"""
        return [
            {
                CandidateEstimatorDF.PARAM_CANDIDATE: [space.estimator.candidate],
                CandidateEstimatorDF.PARAM_CANDIDATE_NAME: [
                    space.estimator.candidate_name
                ],
                **space.get_parameters(),
            }
            for space in self.spaces
        ]

    def to_expression(self) -> "Expression":
        """[see superclass]"""
        # noinspection PyProtectedMember
        return Id(type(self))(
            self.estimator.candidate,
            [
                space._to_expression(path_prefix=CandidateEstimatorDF.PARAM_CANDIDATE)
                for space in self.spaces
            ],
        )


@inheritdoc(match="""[see superclass]""")
class CandidateEstimatorDF(
    ClassifierDF, RegressorDF, TransformerDF, Generic[T_Candidate_co]
):
    """
    Metaclass providing representation for candidate estimator to be used in
    hyperparameter search. Unifies evaluation approach for :class:`.ParameterSpace`
    and class:`.MultiEstimatorParameterSpace`. For the latter it provides "empty"
    candidate where actual estimator is a hyperparameter itself.
    """

    #: name of the `candidate` parameter
    PARAM_CANDIDATE = "candidate"

    #: name of the `candidate_name` parameter
    PARAM_CANDIDATE_NAME = "candidate_name"

    #: The currently selected estimator candidate
    candidate: T_Candidate_co

    #: The name of the candidate
    candidate_name: str

    def __init__(
        self,
        candidate: Optional[T_Candidate_co] = None,
        candidate_name: Optional[str] = None,
    ) -> None:
        """
        :param candidate: the candidate estimator. If ``None`` then estimators to be
                          evaluated should be provided in the parameter grid under a
                          "candidate" key.
        :param candidate_name: a name for the candidate
        """
        super().__init__()

        self.candidate = candidate
        self.candidate_name = candidate_name

    @classmethod
    def empty(cls) -> "CandidateEstimatorDF":
        """
        :return: new candidate instance without internal estimator
        """
        return cls()

    @property
    def classes_(self) -> Sequence[Any]:
        """[see superclass]"""
        return self.candidate.classes_

    # noinspection PyPep8Naming
    def predict_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""
        return self.candidate.predict_proba(X, **predict_params)

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """[see superclass]"""
        return self.candidate.predict_log_proba(X, **predict_params)

    # noinspection PyPep8Naming
    def decision_function(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self.candidate.decision_function(X, **predict_params)

    # noinspection PyPep8Naming
    def score(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None
    ) -> float:
        """[see superclass]"""
        return self.candidate.score(X, y, sample_weight)

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self.candidate.predic(X, **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(
        self, X: pd.DataFrame, y: pd.Series, **fit_params: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """[see superclass]"""
        return self.candidate.fit_predict(X, y, **fit_params)

    # noinspection PyPep8Naming
    def fit(
        self: T_Self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        **fit_params: Any,
    ) -> T_Self:
        """[see superclass]"""
        self.candidate.fit(X, y, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.candidate is not None and self.candidate.is_fitted

    # noinspection PyPep8Naming
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""
        return self.candidate.inverse_transform(X)

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""
        return self.candidate.transform(X)

    @property
    def _estimator_type(self) -> str:
        # noinspection PyProtectedMember
        return self.candidate._estimator_type

    def _get_features_in(self) -> pd.Index:
        return self.candidate.feature_names_in_

    def _get_n_outputs(self) -> int:
        return self.candidate.n_outputs_

    def _get_features_original(self) -> pd.Series:
        return self.candidate.feature_names_original_


__tracker.validate()


#
# auxiliary functions
#


def ensure_subclass(
    estimator_type: Type[T_Candidate_co], expected_type: Type[T_Candidate_co]
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


def validate_spaces(spaces: Collection[ParameterSpace[T_Candidate_co]]) -> None:
    """
    Ensure that all candidates implement the same estimator type (typically regressors
    or classifiers)

    :param spaces: the candidates to check
    """

    estimator_types = {
        getattr(space.estimator, "_estimator_type", None) for space in spaces
    }

    if len(estimator_types) > 1:
        raise TypeError(
            "all candidate estimators must have the same estimator type, "
            "but got multiple types: " + ", ".join(sorted(estimator_types))
        )
