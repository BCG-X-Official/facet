"""
Implementation of :class:`.LearnerInspector`.
"""
import logging
import re
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union, cast

import pandas as pd
from sklearn.base import is_classifier, is_regressor

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
]


#
# Type variables
#

T_SupervisedLearnerDF = TypeVar("T_SupervisedLearnerDF", bound=SupervisedLearnerDF)


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
@subsdoc(
    pattern="Explain a model based on SHAP",
    replacement="Explain a regressor or classifier based on SHAP",
)
@inheritdoc(match="""[see superclass]""")
class LearnerInspector(
    ModelInspector[T_SupervisedLearnerDF], Generic[T_SupervisedLearnerDF]
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

    #: The learner being inspected.
    #:
    #: If the model is a pipeline, this is the final estimator in the pipeline;
    #: otherwise, it is the model itself.
    learner: SupervisedLearnerDF

    # defined in superclass, repeated here for Sphinx:
    model: T_SupervisedLearnerDF
    shap_interaction: bool
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

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
        """
        :param model: the learner pipeline to inspect
        :param explainer_factory: optional function that creates a shap Explainer
            (default: ``TreeExplainerFactory``)
        """

        if not model.is_fitted:
            raise ValueError("arg pipeline must be fitted")

        learner: SupervisedLearnerDF

        if isinstance(model, SupervisedLearnerPipelineDF):
            learner = model.final_estimator
        elif isinstance(model, SupervisedLearnerDF):
            learner = model
        else:
            raise TypeError(
                "arg model must be a SupervisedLearnerPipelineDF or a "
                f"SupervisedLearnerDF, but is a {type(model).__name__}"
            )
        self.learner = learner

        if is_classifier(learner):
            try:
                n_outputs = learner.n_outputs_
            except AttributeError:
                pass
            else:
                if n_outputs > 1:
                    raise ValueError(
                        "only single-target classifiers (binary or multi-class) are "
                        "supported, but the given classifier has been fitted on "
                        f"multiple targets: {', '.join(learner.output_names_)}"
                    )
        elif not is_regressor(learner):
            raise TypeError(
                "learner in arg pipeline must be a classifier or a regressor,"
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
        self._shap_calculator: Optional[LearnerShapCalculator[Any]] = None

    __init__.__doc__ = str(__init__.__doc__) + re.sub(
        r"(?m)^\s*:param model:\s+.*$", "", str(ModelInspector.__init__.__doc__)
    )

    @property
    def feature_names(self) -> List[str]:
        """[see superclass]"""
        return cast(
            List[str],
            self.learner.feature_names_in_.to_list(),
        )

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

    @property
    def shap_calculator(self) -> LearnerShapCalculator[Any]:
        """[see superclass]"""

        if self._shap_calculator is not None:
            return self._shap_calculator

        learner: SupervisedLearnerDF = self.learner

        shap_calculator_params: Dict[str, Any] = dict(
            model=self.learner.native_estimator,
            interaction_values=self.shap_interaction,
            explainer_factory=self.explainer_factory,
            n_jobs=self.n_jobs,
            shared_memory=self.shared_memory,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
        )

        shap_calculator: LearnerShapCalculator[Any]
        if is_classifier(learner):
            shap_calculator = ClassifierShapCalculator(**shap_calculator_params)
        else:
            shap_calculator = RegressorShapCalculator(
                **shap_calculator_params, output_names=learner.output_names_
            )

        self._shap_calculator = shap_calculator
        return shap_calculator


__tracker.validate()
