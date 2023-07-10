"""
Implements the base package.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, Mapping, Optional, TypeVar

import numpy as np
import pandas as pd
from packaging.version import Version
from shap import Explainer, Explanation

from pytools.api import AllTracker
from pytools.expression import HasExpressionRepr

from .._types import ArraysFloat, XType, YType

log = logging.getLogger(__name__)

__all__ = [
    "BaseExplainer",
    "ExplainerFactory",
]


# Apply a hack to address shap's incompatibility with numpy >= 1.24:
# shap relies on the np.bool, np.int, and np.float types, which were deprecated in
# numpy 1.20 and removed in numpy 1.24.
#
# We define these types as an alias for the corresponding type with a trailing
# underscore.

if Version(np.__version__) >= Version("1.20"):
    for __attr in ("bool", "int", "float"):
        setattr(np, __attr, getattr(np, f"{__attr}_"))
    del __attr


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        :param args: positional arguments passed to the explainer constructor
        :param kwargs: keyword arguments passed to the explainer constructor
        """
        super().__init__(*args, **kwargs)

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
        r"""
        Estimate the SHAP interaction values for a set of samples.

        :param X: matrix of samples (# samples x # features) on which to explain the
            model's output
        :param y: array of label values for each sample, used when explaining loss
            functions (optional)
        :param kwargs: additional arguments specific to the explainer implementation
        :return: SHAP values as an array of shape
            :math:`(n_\mathrm{observations}, n_\mathrm{features}, n_\mathrm{features})`;
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
    explainer_kwargs: Dict[str, Any]

    def __init__(self, **explainer_kwargs: Any) -> None:
        """
        :param explainer_kwargs: additional keyword arguments to be passed to the
            explainer
        """
        super().__init__()
        self.explainer_kwargs = explainer_kwargs

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
        :return: the new explainer instance
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
