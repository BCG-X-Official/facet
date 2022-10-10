"""
Implementation of package ``facet.inspection.shap.learner``.
"""

import logging
from abc import ABCMeta
from typing import Any, Generic, List, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

from pytools.api import AllTracker, inheritdoc, subsdoc

from facet.inspection._explainer import (
    BaseExplainer,
    ExplainerFactory,
    ParallelExplainer,
)
from facet.inspection.shap import ShapCalculator

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
T_Learner = TypeVar("T_Learner", bound=Union[RegressorMixin, ClassifierMixin])
T_Regressor = TypeVar("T_Regressor", bound=RegressorMixin)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class LearnerShapCalculator(
    ShapCalculator[T_Learner], Generic[T_Learner], metaclass=ABCMeta
):
    """
    Base class for SHAP calculators based on :mod:`sklearndf` learners.
    """

    #: The supervised learner used to calculate SHAP values.
    learner: T_Learner

    def __init__(
        self,
        learner: T_Learner,
        *,
        explainer_factory: ExplainerFactory[T_Learner],
        interaction_values: bool,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param learner: The supervised learner used to calculate SHAP values.
        """
        super().__init__(
            explainer_factory=explainer_factory,
            interaction_values=interaction_values,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.learner = learner

    __init__.__doc__ = cast(str, __init__.__doc__) + cast(
        str, ShapCalculator.__init__.__doc__
    )

    def validate_features(self, features: pd.DataFrame) -> None:
        """[see superclass]"""

        try:
            features_expected: npt.NDArray[Any] = self.learner.feature_names_in_
        except AttributeError:
            return

        diff = features.columns.symmetric_difference(features_expected)
        if not diff.empty:
            raise ValueError(
                f"Features to be explained do not match the features used to train the"
                f"learner: expected {features_expected.tolist()}, got "
                f"{features.columns.tolist()}."
            )

    def _get_explainer(self, features: pd.DataFrame) -> BaseExplainer:

        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self.explainer_factory.uses_background_dataset:
            background_dataset = features

            background_dataset_not_na = background_dataset.dropna()

            if len(background_dataset_not_na) != len(background_dataset):
                n_original = len(background_dataset)
                n_dropped = n_original - len(background_dataset_not_na)
                log.warning(
                    f"{n_dropped} out of {n_original} observations in the background "
                    f"dataset have missing values after pre-processing and will be "
                    f"dropped."
                )

                background_dataset = background_dataset_not_na

        else:
            background_dataset = None

        learner = self.learner
        explainer = self.explainer_factory.make_explainer(
            model=learner, data=background_dataset
        )

        if self.n_jobs != 1:
            explainer = ParallelExplainer(
                explainer,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )

        return explainer


@inheritdoc(match="""[see superclass]""")
class RegressorShapCalculator(
    LearnerShapCalculator[T_Regressor], Generic[T_Regressor], metaclass=ABCMeta
):
    """
    Calculates SHAP (interaction) values for regression models.
    """

    @subsdoc(
        pattern=r"(?m)(^\s*)(:param learner: .*$)",
        replacement=r"\1\2\n"
        r"\1:param output_names: The names of the outputs of the regressor.",
        using=LearnerShapCalculator.__init__,
    )
    def __init__(
        self,
        learner: T_Learner,
        *,
        output_names: List[str],
        explainer_factory: ExplainerFactory[T_Learner],
        interaction_values: bool,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """[see superclass]"""

        super().__init__(
            learner=learner,
            explainer_factory=explainer_factory,
            interaction_values=interaction_values,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        try:
            n_outputs = learner.n_outputs_
        except AttributeError:
            # assume a single output if the learner lacks the n_outputs_ attribute
            n_outputs = 1

        if n_outputs != len(output_names):
            raise ValueError(
                f"Number of output names ({len(output_names)}) does not match the "
                f"number of outputs of the regressor ({n_outputs})."
            )

        self.output_names = output_names

    #: Multi-output SHAP values are determined by target.
    MULTI_OUTPUT_INDEX_NAME = "target"

    def get_output_names(self) -> List[str]:
        """[see superclass]"""
        return self.output_names

    def _convert_raw_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        if self.interaction_values:
            row_index = pd.MultiIndex.from_product(
                iterables=(observations, features_in_split),
                names=(observations.name, features_in_split.name),
            )

            return [
                pd.DataFrame(
                    data=raw_interaction_tensor.reshape(
                        (-1, raw_interaction_tensor.shape[2])
                    ),
                    index=row_index,
                    columns=features_in_split,
                )
                for raw_interaction_tensor in raw_shap_tensors
            ]
        else:
            return [
                pd.DataFrame(
                    data=raw_shap_matrix, index=observations, columns=features_in_split
                )
                for raw_shap_matrix in raw_shap_tensors
            ]


@inheritdoc(match="""[see superclass]""")
class ClassifierShapCalculator(
    LearnerShapCalculator[T_Classifier], Generic[T_Classifier], metaclass=ABCMeta
):
    """
    Calculates SHAP (interaction) values for classification models.
    """

    #: Multi-output SHAP values are determined by class.
    MULTI_OUTPUT_INDEX_NAME = "class"

    def _convert_shap_tensors_to_list(
        self,
        *,
        shap_tensors: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]],
        n_outputs: int,
    ) -> List[npt.NDArray[np.float_]]:

        # if n_outputs == 2 and isinstance(shap_tensors, np.ndarray):
        #     # if we have a single output *and* binary classification, the explainer
        #     # will have returned a single tensor for the positive class;
        #     # the SHAP values for the negative class will have the opposite sign
        #     (shap_tensors,) = super()._convert_shap_tensors_to_list(
        #         shap_tensors=shap_tensors, n_outputs=1
        #     )
        #     return [-shap_tensors, shap_tensors]
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

    def get_output_names(self) -> List[str]:
        """[see superclass]"""

        classifier = self.learner

        try:
            n_outputs_ = classifier.n_outputs_
        except AttributeError:
            # no n_outputs_ defined; we assume the classifier is not multi-target
            pass
        else:
            assert n_outputs_ == 1, "classification model must be single-output"

        try:
            # noinspection PyTypeChecker
            classes = classifier.classes_
        except AttributeError as cause:
            raise AssertionError("classifier must define classes_ attribute") from cause

        class_names: List[str] = list(map(str, classes))
        n_classes = len(class_names)

        if n_classes == 1:
            raise AssertionError(
                f"cannot explain a model with single class {class_names[0]!r}"
            )

        elif n_classes == 2:
            # for binary classifiers, we will generate only output for the second
            # (positive) class as the probabilities for the second class are trivially
            # linked to class 1
            return class_names[:1]

        else:
            return class_names

    def _convert_raw_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:

        if self.interaction_values:
            # return a list of data frame [(obs x features) x features],
            # one for each of the outputs

            # each row is indexed by an observation and a feature
            row_index = pd.MultiIndex.from_product(
                iterables=(observations, features_in_split),
                names=(observations.name, features_in_split.name),
            )

            return [
                pd.DataFrame(
                    data=raw_shap_interaction_matrix.reshape(
                        (-1, raw_shap_interaction_matrix.shape[2])
                    ),
                    index=row_index,
                    columns=features_in_split,
                )
                for raw_shap_interaction_matrix in raw_shap_tensors
            ]

        else:
            # return a list of data frame [obs x features], one for each of the outputs

            return [
                pd.DataFrame(
                    data=raw_shap_matrix, index=observations, columns=features_in_split
                )
                for raw_shap_matrix in raw_shap_tensors
            ]


__tracker.validate()
