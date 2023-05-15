"""
Factories for SHAP explainers from the ``shap`` package.
"""

import functools
import logging
from abc import ABCMeta, abstractmethod
from multiprocessing.synchronize import Lock as LockType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.base import ClassifierMixin, RegressorMixin
from typing_extensions import TypeAlias

from pytools.api import AllTracker, inheritdoc, validate_type
from pytools.expression import Expression, HasExpressionRepr
from pytools.expression.atomic import Id
from pytools.parallelization import Job, JobQueue, JobRunner, ParallelizableMixin

from ._types import ModelFunction

log = logging.getLogger(__name__)

__all__ = [
    "BaseExplainer",
    "ExactExplainerFactory",
    "ExplainerFactory",
    "ExplainerJob",
    "ExplainerQueue",
    "FunctionExplainerFactory",
    "KernelExplainerFactory",
    "ParallelExplainer",
    "PermutationExplainerFactory",
    "TreeExplainerFactory",
]

#
# conditional and mock imports
#


from shap import Explainer, Explanation

try:
    import catboost
except ImportError:
    from types import ModuleType

    catboost = ModuleType("catboost")
    catboost.Pool = type("Pool", (), {})

#
# Type aliases
#

ArraysAny: TypeAlias = Union[npt.NDArray[Any], List[npt.NDArray[Any]]]
ArraysFloat: TypeAlias = Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]]
Learner: TypeAlias = Union[RegressorMixin, ClassifierMixin]
XType = Union[npt.NDArray[Any], pd.DataFrame, catboost.Pool]
YType = Union[npt.NDArray[Any], pd.Series, None]

#
# Type variables
#

T_Model = TypeVar("T_Model")


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Base classes
#


class BaseExplainer(
    Explainer,  # type: ignore
    metaclass=ABCMeta,
):
    """
    Abstract base class of SHAP explainers, providing stubs for methods used by FACET
    but not consistently supported by class :class:`shap.Explainer` across different
    versions of the `shap` package.

    Provides unified support for the old and new explainer APIs:

    - The old API uses methods :meth:`.shap_values` and :meth:`.shap_interaction_values`
      to compute SHAP values and interaction values, respectively. They return
      *numpy* arrays for single-output or single-class models, and lists of *numpy*
      arrays for multi-output or multi-class models.
    - The new API introduced in :mod:`shap` 0.36 makes explainer objects callable;
      direct calls to an explainer object return an :class:`.Explanation` object
      that contains the SHAP values and interaction values.
      For multi-output or multi-class models, the array has an additional dimension for
      the outputs or classes as the last axis.

    As of :mod:`shap` 0.36, the old API is deprecated for the majority of explainers
    while the :class:`shap.KernelExplainer` still uses the old API exclusively
    in :mod:`shap` 0.41.
    We remedy this by adding support for both APIs to all explainers created through
    an :class:`ExplainerFactory` object.
    """

    @property
    @abstractmethod
    def supports_interaction(self) -> bool:
        """
        ``True`` if the explainer supports interaction effects, ``False`` otherwise.
        """
        pass

    # noinspection PyPep8Naming
    def shap_values(self, X: XType, y: YType = None, **kwargs: Any) -> ArraysFloat:
        """
        Estimate the SHAP values for a set of samples.

        :param X: matrix of samples (# samples x # features) on which to explain the
            model's output
        :param y: array of label values for each sample, used when explaining loss
            functions (optional)
        :param kwargs: additional arguments specific to the explainer implementation
        :return: SHAP values as an array of shape `(n_observations, n_features)`;
            a list of such arrays in the case of a multi-output model
        """

        explanation: Explanation
        if y is None:
            explanation = self(X, **kwargs)
        else:
            explanation = self(X, y, **kwargs)

        values = explanation.values

        interactions: int = kwargs.get("interactions", 1)
        if isinstance(values, np.ndarray):
            if values.ndim == 2 + interactions:
                # convert the array of shape
                # (n_observations, n_features, ..., n_outputs)
                # to a list of arrays of shape (n_observations, n_features, ...)
                return [values[..., i] for i in range(values.shape[-1])]
            elif values.ndim == 1 + interactions:
                # return a single array of shape (n_observations, n_features)
                return values
            else:
                raise ValueError(
                    f"SHAP values have unexpected shape {values.shape}; "
                    "expected shape (n_observations, n_features, ..., n_outputs) "
                    "or (n_observations, n_features, ...)"
                )
        else:
            assert isinstance(values, list), "SHAP values must be a list or array"
            return values

    # noinspection PyPep8Naming,PyUnresolvedReferences
    def shap_interaction_values(
        self, X: XType, y: YType = None, **kwargs: Any
    ) -> ArraysFloat:
        """
        Estimate the SHAP interaction values for a set of samples.

        :param X: matrix of samples (# samples x # features) on which to explain the
            model's output
        :param y: array of label values for each sample, used when explaining loss
            functions (optional)
        :param kwargs: additional arguments specific to the explainer implementation
        :return: SHAP values as an array of shape
            `(n_observations, n_features, n_features)`;
            a list of such arrays in the case of a multi-output model
        """
        if self.supports_interaction:
            return self.shap_values(X, y, interactions=2, **kwargs)
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support interaction values"
            )


class ExplainerFactory(HasExpressionRepr, Generic[T_Model], metaclass=ABCMeta):
    """
    A factory for constructing :class:`~shap.Explainer` objects.
    """

    #: Additional keyword arguments to be passed to the explainer constructor.
    kwargs: Dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        """
        :param kwargs: additional keyword arguments to be passed to the explainer
        """
        super().__init__()
        self.explainer_kwargs = kwargs

    @property
    @abstractmethod
    def explains_raw_output(self) -> bool:
        """
        ``True`` if explainers made by this factory explain raw model output,
        ``False`` otherwise.
        """

    @property
    @abstractmethod
    def supports_shap_interaction_values(self) -> bool:
        """
        ``True`` if explainers made by this factory allow for calculating
        SHAP interaction values, ``False`` otherwise.
        """

    @property
    @abstractmethod
    def uses_background_dataset(self) -> bool:
        """
        ``True`` if explainers made by this factory will use a background dataset
        passed to method :meth:`.make_explainer`, ``False`` otherwise.
        """

    @abstractmethod
    def make_explainer(
        self, model: T_Model, data: Optional[pd.DataFrame]
    ) -> BaseExplainer:
        """
        Construct a new :class:`~shap.Explainer` to compute shap values.

        :param model: fitted learner for which to compute shap values
        :param data: background dataset (optional)
        :return: the new explainer object
        """

    @staticmethod
    def _remove_null_kwargs(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v is not None}

    def _validate_background_dataset(self, data: Optional[pd.DataFrame]) -> None:
        if data is None and self.uses_background_dataset:
            raise ValueError(
                "a background dataset is required to make an explainer with this "
                "factory"
            )


#
# Parallelization support: class ParallelExplainer and helper classes
#


@inheritdoc(match="""[see superclass]""")
class ExplainerJob(Job[ArraysAny]):
    """
    A call to an explainer function with given `X` and `y` values.
    """

    #: the SHAP explainer to use
    explainer: BaseExplainer

    #: if ``False``, calculate SHAp values; otherwise, calculate SHAP interaction values
    interactions: bool

    #: the feature values of the observations to be explained
    X: XType

    #: the target values of the observations to be explained
    y: YType

    #: additional arguments specific to the explainer method
    kwargs: Dict[str, Any]

    # noinspection PyPep8Naming
    def __init__(
        self,
        explainer: BaseExplainer,
        X: XType,
        y: YType = None,
        *,
        interactions: bool,
        **kwargs: Any,
    ) -> None:
        """
        :param explainer: the SHAP explainer to use
        :param X: the feature values of the observations to be explained
        :param y: the target values of the observations to be explained
        :param interactions: if ``False``, calculate SHAP values; if ``True``,
            calculate SHAP interaction values
        :param kwargs: additional arguments specific to the explainer method
        """
        self.explainer = explainer
        self.X = X
        self.y = y
        self.interactions = interactions
        self.kwargs = kwargs

    def run(self) -> ArraysAny:
        """[see superclass]"""

        shap_fn: Callable[..., ArraysAny] = (
            self.explainer.shap_interaction_values
            if self.interactions
            else self.explainer.shap_values
        )

        if self.y is None:
            return shap_fn(self.X, **self.kwargs)
        else:
            return shap_fn(self.X, self.y, **self.kwargs)


@inheritdoc(match="""[see superclass]""")
class ExplainerQueue(JobQueue[ArraysAny, ArraysAny]):
    """
    A queue splitting a data set to be explained into multiple jobs.
    """

    # defined in superclass, repeated here for Sphinx
    lock: LockType

    #: the SHAP explainer to use
    explainer: BaseExplainer

    #: if ``False``, calculate SHAp values; otherwise, calculate SHAP interaction values
    interactions: bool

    #: the feature values of the observations to be explained
    X: npt.NDArray[Any]

    #: the target values of the observations to be explained
    y: Optional[npt.NDArray[Any]]

    #: the maximum number of observations to allocate to each job
    max_job_size: int

    #: additional arguments specific to the explainer method
    kwargs: Dict[str, Any]

    # noinspection PyPep8Naming
    def __init__(
        self,
        explainer: BaseExplainer,
        X: XType,
        y: YType = None,
        *,
        interactions: bool,
        max_job_size: int,
        **kwargs: Any,
    ) -> None:
        """
        :param explainer: the SHAP explainer to use
        :param X: the feature values of the observations to be explained
        :param y: the target values of the observations to be explained
        :param interactions: if ``False``, calculate SHAP values; if ``True``,
            calculate SHAP interaction values
        :param max_job_size: the maximum number of observations to allocate to each job
        :param kwargs: additional arguments specific to the explainer method

        :raise NotImplementedError: if `X` is a :class:`~catboost.Pool`;
            this is currently not supported
        """
        super().__init__()

        if isinstance(X, catboost.Pool):
            raise NotImplementedError("CatBoost Pool is not supported")
        self.explainer = explainer
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.interactions = interactions
        self.max_job_size = max_job_size
        self.kwargs = kwargs

    def jobs(self) -> Iterable[Job[ArraysAny]]:
        """[see superclass]"""

        x = self.X
        y = self.y
        n = len(x)
        job_size = (n - 1) // len(self) + 1
        kwargs = self.kwargs

        return (
            ExplainerJob(
                self.explainer,
                X=x[start : start + job_size].copy(),
                y=None if y is None else y[start : start + job_size].copy(),
                interactions=self.interactions,
                **kwargs,
            )
            for start in range(0, n, job_size)
        )

    def aggregate(self, job_results: List[ArraysAny]) -> ArraysAny:
        """[see superclass]"""
        if isinstance(job_results[0], np.ndarray):
            return np.vstack(job_results)
        else:
            return [np.vstack(arrays) for arrays in zip(*job_results)]

    def __len__(self) -> int:
        return (len(self.X) - 1) // self.max_job_size + 1


@inheritdoc(match="""[see superclass]""")
class ParallelExplainer(BaseExplainer, ParallelizableMixin):
    """
    A wrapper class, turning an explainer into a parallelized version, explaining
    chunks of observations in parallel.
    """

    # defined in superclass, repeated here for Sphinx
    n_jobs: Optional[int]

    # defined in superclass, repeated here for Sphinx
    shared_memory: Optional[bool]

    # defined in superclass, repeated here for Sphinx
    pre_dispatch: Optional[Union[str, int]]

    # defined in superclass, repeated here for Sphinx
    verbose: Optional[int]

    #: The explainer being parallelized by this wrapper
    explainer: BaseExplainer

    #: the maximum number of observations to allocate to any of the explainer jobs
    #: running in parallel
    max_job_size: int

    def __init__(
        self,
        explainer: BaseExplainer,
        *,
        max_job_size: int = 10,
        n_jobs: Optional[int],
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param explainer: the explainer to be parallelized by this wrapper
        :param max_job_size: the maximum number of observations to allocate to any of
            the explainer jobs running in parallel
        """
        Explainer.__init__(self, model=explainer.model)
        ParallelizableMixin.__init__(
            self,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if isinstance(explainer, ParallelExplainer):
            log.warning(
                f"creating parallel explainer from parallel explainer {explainer!r}"
            )

        self.explainer = explainer
        self.max_job_size = max_job_size

    assert __init__.__doc__ is not None
    __init__.__doc__ += cast(str, ParallelizableMixin.__init__.__doc__)

    @property
    def supports_interaction(self) -> bool:
        """[see superclass]"""
        return self.explainer.supports_interaction

    def __call__(self, *args: Any, **kwargs: Any) -> Explanation:
        return self.explainer(*args, **kwargs)

    # noinspection PyPep8Naming
    def shap_values(self, X: XType, y: YType = None, **kwargs: Any) -> ArraysFloat:
        """[see superclass]"""
        if y is None:
            return self.explainer.shap_values(X=X, **kwargs)
        else:
            return self.explainer.shap_values(X=X, y=y, **kwargs)

    # noinspection PyPep8Naming
    def shap_interaction_values(
        self, X: XType, y: YType = None, **kwargs: Any
    ) -> ArraysFloat:
        """[see superclass]"""
        if y is None:
            return self.explainer.shap_interaction_values(X=X, **kwargs)
        else:
            return self.explainer.shap_interaction_values(X=X, y=y, **kwargs)

    # noinspection PyPep8Naming
    def _run(
        self,
        explainer: BaseExplainer,
        X: XType,
        y: YType = None,
        *,
        interactions: bool,
        **kwargs: Any,
    ) -> ArraysFloat:
        return JobRunner.from_parallelizable(self).run_queue(
            ExplainerQueue(
                explainer=explainer,
                X=X,
                y=y,
                interactions=interactions,
                max_job_size=self.max_job_size,
                **kwargs,
            )
        )


#
# TreeExplainer factory
#


@inheritdoc(match="""[see superclass]""")
class _TreeExplainer(
    shap.explainers.Tree,  # type: ignore
    BaseExplainer,
):
    @property
    def supports_interaction(self) -> bool:
        """[see superclass]"""
        return True

    # noinspection PyPep8Naming
    def __call__(
        self, X: XType, y: YType = None, check_additivity: bool = False, **kwargs: Any
    ) -> Explanation:
        # we override the __call__ method to change the default value of
        # arg check_additivity to False
        return cast(
            Explanation,
            super().__call__(X=X, y=y, check_additivity=check_additivity, **kwargs),
        )

    # noinspection PyPep8Naming
    def shap_values(
        self, X: XType, y: YType = None, check_additivity: bool = False, **kwargs: Any
    ) -> ArraysFloat:
        """[see superclass]"""
        return cast(
            ArraysFloat,
            super().shap_values(X=X, y=y, check_additivity=check_additivity, **kwargs),
        )


@inheritdoc(match="""[see superclass]""")
class TreeExplainerFactory(ExplainerFactory[Learner]):
    """
    A factory constructing :class:`~shap.TreeExplainer` instances.
    """

    def __init__(
        self,
        *,
        model_output: Optional[str] = None,
        feature_perturbation: Optional[str] = None,
        uses_background_dataset: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        :param model_output: override the default model output parameter (optional)
        :param feature_perturbation: override the default (optional)
            feature_perturbation parameter
        :param uses_background_dataset: if ``False``, don't pass the background
            dataset on to the tree explainer even if a background dataset is passed
            to :meth:`.make_explainer`
        """
        super().__init__(**kwargs)
        validate_type(
            model_output, expected_type=str, optional=True, name="arg model_output"
        )
        validate_type(
            feature_perturbation,
            expected_type=str,
            optional=True,
            name="arg feature_perturbation",
        )
        self.model_output = model_output
        self.feature_perturbation = feature_perturbation
        self._uses_background_dataset = uses_background_dataset

    __init__.__doc__ = f"{__init__.__doc__}{ExplainerFactory.__init__.__doc__}"

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return self.model_output in [None, "raw"]

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return self.feature_perturbation == "tree_path_dependent"

    @property
    def uses_background_dataset(self) -> bool:
        """[see superclass]"""
        return self._uses_background_dataset

    def make_explainer(
        self, model: Learner, data: Optional[pd.DataFrame] = None
    ) -> BaseExplainer:
        """[see superclass]"""

        self._validate_background_dataset(data=data)

        assert _TreeExplainer is not None, "Global tree explainer is set"

        explainer = _TreeExplainer(
            model=model,
            data=data if self._uses_background_dataset else None,
            **self._remove_null_kwargs(
                dict(
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                )
            ),
            **self.explainer_kwargs,
        )

        return explainer

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return Id(type(self))(
            model_output=self.model_output,
            feature_perturbation=self.feature_perturbation,
            use_background_dataset=self._uses_background_dataset,
        )


#
# Abstract function explainer factory
#


@inheritdoc(match="""[see superclass]""")
class FunctionExplainerFactory(
    ExplainerFactory[Union[Learner, ModelFunction]], metaclass=ABCMeta
):
    """
    A factory constructing :class:`~shap.Explainer` instances that use Python functions
    as the underlying model.
    """

    @property
    def uses_background_dataset(self) -> bool:
        """``True``, since function explainers typically use a background dataset"""
        return True

    def make_explainer(
        self, model: Union[Learner, ModelFunction], data: Optional[pd.DataFrame]
    ) -> BaseExplainer:
        """[see superclass]"""
        self._validate_background_dataset(data=data)

        # create a model function from the model
        try:
            if isinstance(model, RegressorMixin):
                # noinspection PyUnresolvedReferences
                model_fn = model.predict
            elif isinstance(model, ClassifierMixin):
                # noinspection PyUnresolvedReferences
                model_fn = model.predict_proba
            elif callable(model):
                model_fn = model
            else:
                model_fn = None
        except AttributeError as cause:
            raise TypeError(
                f"arg model does not support default prediction method: {cause}"
            ) from cause
        if not model_fn:
            raise TypeError(
                "arg model is neither a regressor nor a classifier: "
                f"{type(model).__name__}"
            )

        return self.make_explainer_from_function(model_fn=model_fn, data=data)

    @abstractmethod
    def make_explainer_from_function(
        self, model_fn: ModelFunction, data: Optional[pd.DataFrame]
    ) -> BaseExplainer:
        """
        Construct an explainer from a function.

        :param model_fn: the function representing the model
        :param data: the background dataset
        :return: the explainer
        """


#
# KernelExplainer factory
#


class _KernelExplainer(
    shap.KernelExplainer,  # type: ignore
    BaseExplainer,
):
    def __call__(self, *args: Any, **kwargs: Any) -> Explanation:
        """[see superclass]"""

        # we override the BaseExplainer implementation because the shap.KernelExplainer
        # implementation does not support __call__
        shap_values = shap.KernelExplainer.shap_values(self, *args, **kwargs)

        if isinstance(shap_values, list):
            # combine the shap values into a single array, along an additional axis
            shap_values = np.stack(shap_values, axis=-1)

        return Explanation(shap_values)

    @property
    def supports_interaction(self) -> bool:
        """
        :return: ``False`` because :class:`~shap.KernelExplainer` does not support
            interaction values
        """
        return False


@inheritdoc(match="""[see superclass]""")
class KernelExplainerFactory(FunctionExplainerFactory):
    """
    A factory constructing :class:`~shap.KernelExplainer` instances.
    """

    def __init__(
        self,
        *,
        link: Optional[str] = None,
        l1_reg: Optional[str] = "num_features(10)",
        data_size_limit: Optional[int] = 100,
        **kwargs: Any,
    ) -> None:
        """
        :param link: override the default link parameter (optional)
        :param l1_reg: override the default l1_reg parameter of method
            :meth:`~shap.KernelExplainer.shap_values`; pass ``None`` to use the
            default value used by :meth:`~shap.KernelExplainer.shap_values` (optional)
        :param data_size_limit: maximum number of observations to use as
            the background data set; larger data sets will be down-sampled using
            kmeans; don't downsample if omitted (optional)
        """
        super().__init__(**kwargs)
        validate_type(link, expected_type=str, optional=True, name="arg link")
        self.link = link
        self.l1_reg = l1_reg if l1_reg is not None else "num_features(10)"
        self.data_size_limit = data_size_limit

    __init__.__doc__ = f"{__init__.__doc__}{FunctionExplainerFactory.__init__.__doc__}"

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return self.link in [None, "identity"]

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return False

    def make_explainer_from_function(
        self, model_fn: ModelFunction, data: Optional[pd.DataFrame]
    ) -> BaseExplainer:
        """[see superclass]"""

        self._validate_background_dataset(data=data)
        assert data is not None, "this explainer requires a background dataset"

        data_size_limit = self.data_size_limit
        if data_size_limit is not None and len(data) > data_size_limit:
            data = shap.kmeans(data, data_size_limit, round_values=True)

        explainer = _KernelExplainer(
            model=model_fn,
            data=data,
            **self._remove_null_kwargs(dict(link=self.link)),
            **self.explainer_kwargs,
        )

        if self.l1_reg is not None:
            # mypy - disabling type check due to method assignment
            explainer.shap_values = functools.partial(  # type: ignore
                explainer.shap_values, l1_reg=self.l1_reg
            )

        return explainer

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return Id(type(self))(
            link=self.link,
            l1_reg=self.l1_reg,
            data_size_limit=self.data_size_limit,
        )


#
# Exact explainer factory
#

# noinspection PyPep8Naming
@inheritdoc(match="""[see superclass]""")
class _ExactExplainer(
    shap.explainers.Exact,  # type: ignore
    BaseExplainer,
):
    @property
    def supports_interaction(self) -> bool:
        """
        :return: ``True`` because :class:`~shap.explainers.Exact` supports interaction
            values
        """
        return True


@inheritdoc(match="""[see superclass]""")
class ExactExplainerFactory(FunctionExplainerFactory):
    """
    A factory constructing :class:`~shap.Exact` explainer instances.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return True

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return True

    def make_explainer_from_function(
        self, model_fn: ModelFunction, data: Optional[pd.DataFrame]
    ) -> BaseExplainer:
        """[see superclass]"""
        self._validate_background_dataset(data=data)
        # noinspection PyTypeChecker
        return _ExactExplainer(model=model_fn, masker=data)

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return Id(type(self))()


#
# Permutation explainer factory
#


@inheritdoc(match="""[see superclass]""")
# noinspection PyPep8Naming
class _PermutationExplainer(
    shap.explainers.Permutation,  # type: ignore
    BaseExplainer,
):
    @property
    def supports_interaction(self) -> bool:
        """
        :return: ``False`` because :class:`~shap.explainers.Permutation` does not
            support interaction values
        """
        return False

    # noinspection PyPep8Naming
    def shap_values(self, X: XType, y: YType = None, **kwargs: Any) -> ArraysFloat:
        # skip the call to super().shap_values() because would raise
        # an AttributeError exception due to a bug in the shap library
        return BaseExplainer.shap_values(self, X, y, **kwargs)


@inheritdoc(match="""[see superclass]""")
class PermutationExplainerFactory(FunctionExplainerFactory):
    """
    A factory constructing :class:`~shap.Permutation` explainer instances.
    """

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return True

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return False

    def make_explainer_from_function(
        self,
        model_fn: ModelFunction,
        data: Optional[pd.DataFrame],
    ) -> BaseExplainer:
        """[see superclass]"""
        self._validate_background_dataset(data=data)
        # noinspection PyTypeChecker
        return _PermutationExplainer(
            model=model_fn, masker=data, **self.explainer_kwargs
        )

    def to_expression(self) -> Expression:
        """[see superclass]"""
        return Id(type(self))()


__tracker.validate()
