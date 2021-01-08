"""
Factories for SHAP explainers from the ``shap`` package.
"""

import functools
import logging
from abc import ABCMeta, abstractmethod
from distutils import version
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import shap
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from pytools.api import AllTracker, inheritdoc, validate_type
from sklearndf import ClassifierDF, LearnerDF, RegressorDF

log = logging.getLogger(__name__)

__all__ = ["ExplainerFactory", "TreeExplainerFactory", "KernelExplainerFactory"]

#
# Ensure we have a supported version of the shap package
#

_SHAP_EARLIEST_VERSION = version.LooseVersion("0.34")

if version.LooseVersion(shap.__version__) < _SHAP_EARLIEST_VERSION:
    raise RuntimeError(
        f"shap package version {shap.__version__} is not supported; "
        f"please upgrade to version {_SHAP_EARLIEST_VERSION} or later"
    )

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class ExplainerFactory(metaclass=ABCMeta):
    """
    A factory for constructing :class:`~shap.Explainer` objects.
    """

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
        self, model: LearnerDF, data: Optional[pd.DataFrame]
    ) -> Explainer:
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


@inheritdoc(match="[see superclass]")
class TreeExplainerFactory(ExplainerFactory):
    """
    A factory constructing :class:`~shap.TreeExplainer` objects.
    """

    def __init__(
        self,
        model_output: Optional[str] = None,
        feature_perturbation: Optional[str] = None,
        use_background_dataset: bool = True,
    ) -> None:
        """
        :param model_output: (optional) override the default model output parameter
        :param feature_perturbation: (optional) override the default
            feature_perturbation parameter
        :param use_background_dataset: if ``False``, don't pass the background
            dataset on to the tree explainer even if a background dataset is passed
            to :meth:`.make_explainer`
        """
        super().__init__()
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
        self._uses_background_dataset = use_background_dataset

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
        self, model: LearnerDF, data: Optional[pd.DataFrame] = None
    ) -> Explainer:
        """[see superclass]"""

        self._validate_background_dataset(data=data)

        explainer = shap.TreeExplainer(
            model=model.native_estimator,
            data=data if self._uses_background_dataset else None,
            **self._remove_null_kwargs(
                dict(
                    model_output=self.model_output,
                    feature_perturbation=self.feature_perturbation,
                )
            ),
        )

        explainer.shap_values = functools.partial(
            explainer.shap_values, check_additivity=False
        )

        return explainer


@inheritdoc(match="[see superclass]")
class KernelExplainerFactory(ExplainerFactory):
    """
    A factory constructing :class:`~shap.KernelExplainer` objects.
    """

    def __init__(
        self,
        link: Optional[str] = None,
        l1_reg: Optional[str] = "num_features(10)",
        data_size_limit: Optional[int] = 100,
    ) -> None:
        """
        :param link: (optional) override the default link parameter
        :param l1_reg: (optional) override the default l1_reg parameter of method
            :meth:`~shap.KernelExplainer.shap_values`; pass ``None`` to use the
            default value used by :meth:`~shap.KernelExplainer.shap_values`
        :param data_size_limit: (optional) maximum number of observations to use as
            the background data set; larger data sets will be down-sampled using
            kmeans.
            Pass ``None`` to prevent down-sampling the background data set
        """
        super().__init__()
        validate_type(link, expected_type=str, optional=True, name="arg link")
        self.link = link
        self.l1_reg = l1_reg if l1_reg is not None else "num_features(10)"
        self.data_size_limit = data_size_limit

    @property
    def explains_raw_output(self) -> bool:
        """[see superclass]"""
        return self.link in [None, "identity"]

    @property
    def supports_shap_interaction_values(self) -> bool:
        """[see superclass]"""
        return False

    @property
    def uses_background_dataset(self) -> bool:
        """[see superclass]"""
        return True

    def make_explainer(self, model: LearnerDF, data: pd.DataFrame) -> Explainer:
        """[see superclass]"""

        self._validate_background_dataset(data=data)

        model_root_estimator: BaseEstimator = model.native_estimator

        try:
            if isinstance(model, RegressorDF):
                # noinspection PyUnresolvedReferences
                model_fn = model_root_estimator.predict
            elif isinstance(model, ClassifierDF):
                # noinspection PyUnresolvedReferences
                model_fn = model_root_estimator.predict_proba
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

        data_size_limit = self.data_size_limit
        if data_size_limit is not None and len(data) > data_size_limit:
            data = shap.kmeans(data, data_size_limit, round_values=True)

        explainer = shap.KernelExplainer(
            model=model_fn, data=data, **self._remove_null_kwargs(dict(link=self.link))
        )

        if self.l1_reg is not None:
            explainer.shap_values = functools.partial(
                explainer.shap_values, l1_reg=self.l1_reg
            )

        return explainer


__tracker.validate()
