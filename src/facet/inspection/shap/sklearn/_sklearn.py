"""
Implementation of package ``facet.inspection.shap.learner``.
"""

import logging
from abc import ABCMeta
from typing import Generic, List, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin, is_classifier, is_regressor

from pytools.api import AllTracker, inheritdoc, subsdoc

from ...._types import NativeSupervisedLearner
from ....explanation.base import ExplainerFactory
from .. import ShapCalculator

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierShapCalculator",
    "LearnerShapCalculator",
    "RegressorShapCalculator",
]

#
# Type variables
#

T_Classifier = TypeVar("T_Classifier", bound=ClassifierMixin)
T_Learner = TypeVar("T_Learner", bound=NativeSupervisedLearner)
T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class LearnerShapCalculator(
    ShapCalculator[T_Learner], Generic[T_Learner], metaclass=ABCMeta
):
    """
    Base class for SHAP calculators based on :mod:`sklearndf` supervised learners.
    """

    # defined in superclass, repeated here for Sphinx:
    model: T_Learner
    explainer_factory: ExplainerFactory[T_Learner]
    shap_: Optional[pd.DataFrame]
    feature_index_: Optional[pd.Index]
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

    @property
    def input_names(self) -> Optional[List[str]]:
        try:
            return cast(List[str], self.model.feature_names_in_.tolist())
        except AttributeError:
            # the learner does not have a feature_names_in_ attribute,
            return None


@inheritdoc(match="""[see superclass]""")
class RegressorShapCalculator(
    LearnerShapCalculator[T_Regressor], Generic[T_Regressor], metaclass=ABCMeta
):
    """
    Calculates SHAP (interaction) values for regression models.
    """

    # defined in superclass, repeated here for Sphinx:
    model: T_Regressor
    explainer_factory: ExplainerFactory[T_Regressor]
    shap_: Optional[pd.DataFrame]
    feature_index_: Optional[pd.Index]
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

    @subsdoc(
        pattern=r"(?m)(^\s*)(:param model: .*$)",
        replacement=r"\1\2\n"
        r"\1:param output_names: the names of the outputs of the regressor",
        using=LearnerShapCalculator.__init__,
    )
    def __init__(
        self,
        model: T_Regressor,
        *,
        output_names: List[str],
        explainer_factory: ExplainerFactory[T_Regressor],
        interaction_values: bool,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """[see superclass]"""

        super().__init__(
            model=model,
            explainer_factory=explainer_factory,
            interaction_values=interaction_values,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if not is_regressor(model):
            raise ValueError(
                f"regressor SHAP calculator requires a regressor, "
                f"but got a {type(model)}"
            )

        try:
            n_outputs = model.n_outputs_
        except AttributeError:
            # assume a single output if the learner lacks the n_outputs_ attribute
            n_outputs = 1

        if n_outputs != len(output_names):
            raise ValueError(
                f"Number of output names ({len(output_names)}) does not match the "
                f"number of outputs of the regressor ({n_outputs})."
            )

        self._output_names = output_names

    #: Multi-output SHAP values are determined by target.
    MULTI_OUTPUT_INDEX_NAME = "target"

    @property
    def output_names(self) -> List[str]:
        """[see superclass]"""
        return self._output_names

    def _convert_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observation_idx: pd.Index,
        feature_idx: pd.Index,
    ) -> List[pd.DataFrame]:
        # Convert shap tensors to data frames.

        return self._convert_raw_shap_to_df(
            raw_shap_tensors=raw_shap_tensors,
            observation_idx=observation_idx,
            feature_idx=feature_idx,
        )


@inheritdoc(match="""[see superclass]""")
class ClassifierShapCalculator(
    LearnerShapCalculator[T_Classifier], Generic[T_Classifier], metaclass=ABCMeta
):
    """
    Calculates SHAP (interaction) values for classification models.
    """

    #: Multi-output SHAP values are determined by class.
    MULTI_OUTPUT_INDEX_NAME = "class"

    # defined in superclass, repeated here for Sphinx:
    model: T_Classifier
    explainer_factory: ExplainerFactory[T_Classifier]
    shap_: Optional[pd.DataFrame]
    feature_index_: Optional[pd.Index]
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

    def __init__(
        self,
        model: T_Classifier,
        *,
        explainer_factory: ExplainerFactory[T_Classifier],
        interaction_values: bool,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """[see superclass]"""
        super().__init__(
            model=model,
            explainer_factory=explainer_factory,
            interaction_values=interaction_values,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        if not is_classifier(model):
            raise ValueError(
                f"classifier SHAP calculator requires a classifier, "
                f"but got a {type(model)}"
            )

        self._output_names = classifier_shap_output_names(model)

    @property
    def output_names(self) -> List[str]:
        """[see superclass]"""
        return self._output_names

    @staticmethod
    def validate_learner(learner: T_Classifier) -> None:
        """[see superclass]"""

        try:
            n_outputs_ = learner.n_outputs_
        except AttributeError:
            # no n_outputs_ defined; we assume the classifier is not multi-target
            pass
        else:
            if n_outputs_ > 1:
                raise ValueError(
                    "classifier SHAP calculator does not support multi-output "
                    "classifiers, but got a classifier with n_outputs_="
                    f"{n_outputs_}"
                )

    def _convert_shap_tensors_to_list(
        self,
        *,
        shap_tensors: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]],
        n_outputs: int,
    ) -> List[npt.NDArray[np.float_]]:
        if n_outputs == 1 and isinstance(shap_tensors, list) and len(shap_tensors) == 2:
            # in the binary classification case, we will proceed with SHAP values
            # for class 0 only, since values for class 1 will just be the same
            # values times (*-1)  (the opposite delta probability)

            # to ensure the values are returned as expected above,
            # and no information of class 1 is discarded, assert the
            # following:
            if not np.allclose(shap_tensors[0], -shap_tensors[1]):
                _raw_shap_tensor_totals = shap_tensors[0] + shap_tensors[1]
                log.warning(
                    "shap values of binary classifiers should add up to 0.0 "
                    "for each observation and feature, but total shap values range "
                    f"from {_raw_shap_tensor_totals.min():g} "
                    f"to {_raw_shap_tensor_totals.max():g}"
                )

            return super()._convert_shap_tensors_to_list(
                shap_tensors=shap_tensors[1], n_outputs=1
            )
        else:
            return super()._convert_shap_tensors_to_list(
                shap_tensors=shap_tensors, n_outputs=n_outputs
            )

    def _convert_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observation_idx: pd.Index,
        feature_idx: pd.Index,
    ) -> List[pd.DataFrame]:
        if self.interaction_values:
            # return a list of data frame [(obs x features) x features],
            # one for each of the outputs

            # each row is indexed by an observation and a feature
            row_index = pd.MultiIndex.from_product(
                iterables=(observation_idx, feature_idx),
                names=(observation_idx.name, feature_idx.name),
            )

            return [
                pd.DataFrame(
                    data=raw_shap_interaction_matrix.reshape(
                        (-1, raw_shap_interaction_matrix.shape[2])
                    ),
                    index=row_index,
                    columns=feature_idx,
                )
                for raw_shap_interaction_matrix in raw_shap_tensors
            ]

        else:
            # return a list of data frame [obs x features], one for each of the outputs

            return [
                pd.DataFrame(
                    data=raw_shap_matrix, index=observation_idx, columns=feature_idx
                )
                for raw_shap_matrix in raw_shap_tensors
            ]


__tracker.validate()


#
# auxiliary methods
#


def classifier_shap_output_names(classifier: ClassifierMixin) -> List[str]:
    """
    Get the names of the SHAP outputs that will be generated for the given classifier.

    For binary classifiers, the only output name is the name of the positive class.
    For multi-class classifiers, the output names are the names of all classes.

    The classifier must be fitted, and must have a ``classes_`` attribute.

    :param classifier: a classifier
    :return: the names of the SHAP outputs
    :raises ValueError: if the classifier does not define the ``classes_`` attribute,
        is multi-output, or has only a single class
    """
    try:
        # noinspection PyUnresolvedReferences
        classes = classifier.classes_
    except AttributeError as cause:
        raise ValueError("classifier must define classes_ attribute") from cause

    if not isinstance(classes, np.ndarray):
        raise ValueError(
            "classifier must be single-output, with classes_ as a numpy array"
        )

    class_names: List[str] = list(map(str, classes))
    n_classes = len(class_names)

    if n_classes == 1:
        raise ValueError(f"cannot explain a model with single class {class_names[0]!r}")

    elif n_classes == 2:
        # for binary classifiers, we will generate only output for the second
        # (positive) class as the probabilities for the second class are trivially
        # linked to class 1
        return class_names[:1]

    else:
        return class_names
