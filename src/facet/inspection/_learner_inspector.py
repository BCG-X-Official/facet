"""
Implementation of :class:`.LearnerInspector`.
"""
import logging
import re
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, cast

import pandas as pd
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.pipeline import Pipeline

from pytools.api import AllTracker, inheritdoc, subsdoc
from sklearndf import SupervisedLearnerDF
from sklearndf.pipeline import SupervisedLearnerPipelineDF

from .._types import NativeSupervisedLearner
from ..explanation import TreeExplainerFactory
from ..explanation.base import ExplainerFactory
from .base import ModelInspector
from .shap.sklearn import (
    ClassifierShapCalculator,
    LearnerShapCalculator,
    RegressorShapCalculator,
)

log = logging.getLogger(__name__)

__all__ = [
    "LearnerInspector",
    "NativeLearnerInspector",
]


#
# Type variables
#

T_SupervisedLearnerDF = TypeVar("T_SupervisedLearnerDF", bound=SupervisedLearnerDF)
T_SupervisedLearner = TypeVar(
    "T_SupervisedLearner", bound=Union[NativeSupervisedLearner, Pipeline]
)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@subsdoc(
    pattern=(
        r"\n( *)\.\. note:: *\n"  # .. note:: at start of line
        r"(?:\1.*\S\n)+"  # followed by one or more indented lines
        r"(?: *\n)*"  # followed by zero or more blank lines
    ),
    replacement="\n\n",
)
@inheritdoc(match="""[see superclass]""")
class _BaseLearnerInspector(
    ModelInspector[T_SupervisedLearner], Generic[T_SupervisedLearner], metaclass=ABCMeta
):
    """[see superclass]"""

    #: The default explainer factory used by this inspector.
    #: This is a tree explainer using the tree_path_dependent method for
    #: feature perturbation, so we can calculate SHAP interaction values.
    DEFAULT_EXPLAINER_FACTORY = TreeExplainerFactory(
        feature_perturbation="tree_path_dependent", uses_background_dataset=False
    )

    #: The factory instance used to create the explainer for the learner.
    explainer_factory: ExplainerFactory[NativeSupervisedLearner]

    #: the supervised learner to inspect; this is either identical with
    #: :attr:`model`, or the final estimator of :attr:`model` if :attr:`model`
    #: is a pipeline
    learner: NativeSupervisedLearner

    # the SHAP calculator used by this inspector
    _shap_calculator: Optional[LearnerShapCalculator[Any]]

    def __init__(
        self,
        model: T_SupervisedLearner,
        *,
        explainer_factory: Optional[ExplainerFactory[NativeSupervisedLearner]] = None,
        shap_interaction: bool = True,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param model: the learner or learner pipeline to inspect
        :param explainer_factory: optional function that creates a shap Explainer
            (default: ``TreeExplainerFactory``)
        """

        fitted = self._is_model_fitted(model)
        if not fitted:
            raise ValueError("arg model must be fitted")

        learner = self._get_learner(model)

        if is_classifier(learner):
            try:
                # noinspection PyUnresolvedReferences
                n_outputs = learner.n_outputs_
            except AttributeError:
                pass
            else:
                if n_outputs > 1:
                    raise ValueError(
                        "only single-target classifiers (binary or multi-class) are "
                        "supported, but the given classifier has been fitted on "
                        f"multiple targets: {', '.join(model.output_names_)}"
                    )
        elif not is_regressor(learner):
            raise TypeError(
                "learner in arg model must be a classifier or a regressor,"
                f"but is a {type(learner).__name__}"
            )

        if explainer_factory:
            if not explainer_factory.explains_raw_output:
                raise ValueError(
                    "arg explainer_factory is not configured to explain raw output"
                )
        else:
            explainer_factory = self.DEFAULT_EXPLAINER_FACTORY
            assert explainer_factory.explains_raw_output

        if shap_interaction:
            if not explainer_factory.supports_shap_interaction_values:
                log.warning(
                    "ignoring arg shap_interaction=True: "
                    f"explainers made by {explainer_factory!r} do not support "
                    "SHAP interaction values"
                )
                shap_interaction = False

        super().__init__(
            model=model,
            shap_interaction=shap_interaction,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self.explainer_factory = explainer_factory
        self.learner = learner
        self._shap_calculator: Optional[LearnerShapCalculator[Any]] = None

    __init__.__doc__ = str(__init__.__doc__) + re.sub(
        r"(?m)^\s*:param model:\s+.*$", "", str(ModelInspector.__init__.__doc__)
    )

    @property
    @abstractmethod
    def native_learner(self) -> NativeSupervisedLearner:
        """
        The native learner to inspect.
        """

    @property
    def feature_names(self) -> List[str]:
        """[see superclass]"""
        # noinspection PyUnresolvedReferences
        return cast(
            List[str],
            # feature_names_in_ is a pandas index (sklearndf) or an ndarray (sklearn);
            # we convert it to a list
            self.learner.feature_names_in_.tolist(),
        )

    @property
    def shap_calculator(self) -> LearnerShapCalculator[Any]:
        """[see superclass]"""

        if self._shap_calculator is not None:
            return self._shap_calculator

        native_learner = self.native_learner

        shap_calculator_params: Dict[str, Any] = dict(
            model=native_learner,
            interaction_values=self.shap_interaction,
            explainer_factory=self.explainer_factory,
            n_jobs=self.n_jobs,
            shared_memory=self.shared_memory,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
        )

        shap_calculator: LearnerShapCalculator[Any]
        if is_classifier(native_learner):
            shap_calculator = ClassifierShapCalculator(**shap_calculator_params)
        else:
            shap_calculator = RegressorShapCalculator(
                **shap_calculator_params, output_names=self._learner_output_names
            )

        self._shap_calculator = shap_calculator
        return shap_calculator

    @property
    @abstractmethod
    def _learner_output_names(self) -> List[str]:
        """
        The names of the outputs of the learner.
        """
        pass

    @staticmethod
    @abstractmethod
    def _is_model_fitted(model: T_SupervisedLearner) -> bool:
        # return True if the model is fitted, False otherwise
        pass

    @staticmethod
    @abstractmethod
    def _get_learner(model: T_SupervisedLearner) -> NativeSupervisedLearner:
        # get the learner class from the model, which may be a pipeline
        # that includes additional preprocessing steps
        pass


@subsdoc(
    pattern=r"Explain a model",
    replacement=r"Explain an :mod:`sklearndf` regressor or classifier",
)
@inheritdoc(match="""[see superclass]""")
class LearnerInspector(
    _BaseLearnerInspector[T_SupervisedLearnerDF], Generic[T_SupervisedLearnerDF]
):
    """[see superclass]"""

    # defined in superclass, repeated here for Sphinx:
    model: T_SupervisedLearnerDF
    shap_interaction: bool
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]
    explainer_factory: ExplainerFactory[NativeSupervisedLearner]
    learner: SupervisedLearnerDF

    @subsdoc(
        pattern=r"(?m)^(\s*:param model:\s+.*)$",
        replacement=r"""\1 (typically, one of
    a :class:`~sklearndf.pipeline.ClassifierPipelineDF`,
    :class:`~sklearndf.pipeline.RegressorPipelineDF`,
    :class:`~sklearndf.classification.ClassifierDF`, or
    :class:`~sklearndf.regression.RegressorDF`)""",
        using=_BaseLearnerInspector.__init__,
    )
    def __init__(
        self,
        model: T_SupervisedLearnerDF,
        *,
        explainer_factory: Optional[ExplainerFactory[NativeSupervisedLearner]] = None,
        shap_interaction: bool = True,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            explainer_factory=explainer_factory,
            shap_interaction=shap_interaction,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

    @property
    def native_learner(self) -> NativeSupervisedLearner:
        """[see superclass]"""
        return cast(NativeSupervisedLearner, self.learner.native_estimator)

    @property
    def _learner_output_names(self) -> List[str]:
        """[see superclass]"""
        return self.learner.output_names_

    def preprocess_features(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """[see superclass]"""
        if self.model is self.learner:
            # we have a simple learner: no preprocessing needed
            return features
        else:
            # we have a pipeline: preprocess features
            return self.model.preprocess(features)

    @staticmethod
    def _is_model_fitted(model: T_SupervisedLearnerDF) -> bool:
        return model.is_fitted

    @staticmethod
    def _get_learner(model: T_SupervisedLearnerDF) -> SupervisedLearnerDF:
        if isinstance(model, SupervisedLearnerPipelineDF):
            return cast(SupervisedLearnerDF, model.final_estimator)
        elif isinstance(model, SupervisedLearnerDF):
            return model
        else:
            raise TypeError(
                "arg model must be a SupervisedLearnerPipelineDF or a "
                f"SupervisedLearnerDF, but is a {type(model).__name__}"
            )


@subsdoc(
    pattern=r"Explain a model",
    replacement=r"Explain a native scikit-learn regressor or classifier",
)
@inheritdoc(match="""[see superclass]""")
class NativeLearnerInspector(
    _BaseLearnerInspector[T_SupervisedLearner], Generic[T_SupervisedLearner]
):
    """[see superclass]"""

    #: The default explainer factory used by this inspector.
    #: This is a tree explainer using the tree_path_dependent method for
    #: feature perturbation, so we can calculate SHAP interaction values.
    DEFAULT_EXPLAINER_FACTORY = TreeExplainerFactory(
        feature_perturbation="tree_path_dependent", uses_background_dataset=False
    )

    # defined in superclass, repeated here for Sphinx:
    model: T_SupervisedLearner
    shap_interaction: bool
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]
    explainer_factory: ExplainerFactory[NativeSupervisedLearner]
    learner: NativeSupervisedLearner

    @subsdoc(
        pattern=r"(?m)^(\s*:param model:\s+.*)$",
        replacement=r"""\1 (either a scikit-learn :class:`~sklearn.pipeline.Pipeline`,
        or a regressor or classifier that implements the scikit-learn API)""",
        using=_BaseLearnerInspector.__init__,
    )
    def __init__(
        self,
        model: T_SupervisedLearner,
        *,
        explainer_factory: Optional[ExplainerFactory[NativeSupervisedLearner]] = None,
        shap_interaction: bool = True,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            explainer_factory=explainer_factory,
            shap_interaction=shap_interaction,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

    @property
    def native_learner(self) -> NativeSupervisedLearner:
        return self.learner

    @property
    def _learner_output_names(self) -> List[str]:
        # we try to get the number of outputs from the learner; if that fails,
        # we assume that the learner was fitted on a single target
        n_outputs = getattr(self.learner, "n_outputs_", 1)
        if n_outputs == 1:
            return ["y"]
        else:
            return [f"y_{i}" for i in range(n_outputs)]

    def preprocess_features(
        self, features: Union[pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """[see superclass]"""
        if self.learner is self.model:
            # we have a single learner: do not preprocess
            return features
        else:
            # we have a pipeline: preprocessing is the first part of the pipeline
            preprocessing = self.model[:-1]
            return pd.DataFrame(
                preprocessing.transform(features),
                index=features.index,
                columns=preprocessing.get_feature_names_out(),
            )

    @staticmethod
    def _is_model_fitted(model: T_SupervisedLearner) -> bool:
        return is_fitted(model)

    @staticmethod
    def _get_learner(model: T_SupervisedLearner) -> NativeSupervisedLearner:
        if isinstance(model, Pipeline):
            try:
                return model[-1]
            except IndexError:
                raise ValueError("arg model is an empty pipeline")
        else:
            return model


__tracker.validate()


#
# Private auxiliary methods
#


def is_fitted(estimator: BaseEstimator) -> bool:
    """
    Check if the estimator is fitted.

    :param estimator: a scikit-learn estimator instance
    :return: ``True`` if the estimator is fitted; ``False`` otherwise
    """

    if not isinstance(estimator, BaseEstimator):
        raise TypeError(
            "arg estimator must be a scikit-learn estimator, but is a "
            f"{type(estimator).__name__}"
        )

    # get all properties of the estimator (instances of class ``property``)
    fitted_properties = {
        name
        for cls in reversed(type(estimator).mro())
        # traverse the class hierarchy in reverse order, so that we add the
        # properties of the most specific class last
        for name, value in vars(cls).items()
        if (
            # we're only interested in properties that scikit-learn
            # sets when fitting a learner
            name.endswith("_")
            and not name.startswith("_")
            and isinstance(value, property)
        )
    }

    # get all attributes ending with an underscore - these are only set as an estimator
    # is fitted
    fitted_attributes = [
        name
        for name in vars(estimator)
        if name not in fitted_properties
        and name.endswith("_")
        and not name.startswith("_")
    ]

    if fitted_attributes:
        # we have at least one fitted attribute: the estimator is fitted
        return True

    # ensure that at least one of the fitted properties is defined
    for p in fitted_properties:
        if hasattr(estimator, p):
            return True

    # the estimator has no fitted attributes and no fitted properties:
    # it is not fitted
    return False
