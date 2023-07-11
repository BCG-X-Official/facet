"""
Factories for SHAP explainers from the ``shap`` package.
"""

import functools
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union, cast

import numpy as np
import pandas as pd
import shap
from shap import Explanation
from sklearn.base import ClassifierMixin, RegressorMixin

from pytools.api import AllTracker, inheritdoc, validate_type
from pytools.expression import Expression
from pytools.expression.atomic import Id

from .._types import ModelFunction, NativeSupervisedLearner
from ._types import ArraysFloat, XType, YType
from .base import BaseExplainer, ExplainerFactory

log = logging.getLogger(__name__)

__all__ = [
    "ExactExplainerFactory",
    "FunctionExplainerFactory",
    "KernelExplainerFactory",
    "PermutationExplainerFactory",
    "TreeExplainerFactory",
]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


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
class TreeExplainerFactory(ExplainerFactory[NativeSupervisedLearner]):
    """
    A factory constructing :class:`~shap.TreeExplainer` instances.
    """

    # defined in superclass, repeated here for Sphinx
    explainer_kwargs: Dict[str, Any]

    def __init__(
        self,
        *,
        model_output: Optional[str] = None,
        feature_perturbation: Optional[str] = None,
        uses_background_dataset: bool = True,
        **explainer_kwargs: Any,
    ) -> None:
        """
        :param model_output: override the default model output parameter (optional)
        :param feature_perturbation: override the default (optional)
            feature_perturbation parameter
        :param uses_background_dataset: if ``False``, don't pass the background
            dataset on to the tree explainer even if a background dataset is passed
            to :meth:`.make_explainer`
        """
        super().__init__(**explainer_kwargs)
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
        self, model: NativeSupervisedLearner, data: Optional[pd.DataFrame] = None
    ) -> BaseExplainer:
        """[see superclass]"""

        self._validate_background_dataset(data=data)

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
    ExplainerFactory[Union[NativeSupervisedLearner, ModelFunction]], metaclass=ABCMeta
):
    """
    A factory constructing :class:`~shap.Explainer` instances that use Python functions
    as the underlying model.
    """

    # defined in superclass, repeated here for Sphinx
    explainer_kwargs: Dict[str, Any]

    @property
    def uses_background_dataset(self) -> bool:
        """``True``, since function explainers typically use a background dataset"""
        return True

    def make_explainer(
        self,
        model: Union[NativeSupervisedLearner, ModelFunction],
        data: Optional[pd.DataFrame],
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

    # defined in superclass, repeated here for Sphinx
    explainer_kwargs: Dict[str, Any]

    def __init__(
        self,
        *,
        link: Optional[str] = None,
        l1_reg: Optional[str] = "num_features(10)",
        data_size_limit: Optional[int] = 100,
        **explainer_kwargs: Any,
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
        super().__init__(**explainer_kwargs)
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

    # defined in superclass, repeated here for Sphinx
    explainer_kwargs: Dict[str, Any]

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
        """[see superclass]"""
        # skip the call to super().shap_values() because would raise
        # an AttributeError exception due to a bug in the shap library
        return BaseExplainer.shap_values(self, X, y, **kwargs)


@inheritdoc(match="""[see superclass]""")
class PermutationExplainerFactory(FunctionExplainerFactory):
    """
    A factory constructing :class:`~shap.Permutation` explainer instances.
    """

    # defined in superclass, repeated here for Sphinx
    explainer_kwargs: Dict[str, Any]

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
