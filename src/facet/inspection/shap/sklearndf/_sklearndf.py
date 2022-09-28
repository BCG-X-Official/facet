"""
Implementation of package ``facet.inspection.shap.learner``.
"""

import logging
from abc import ABCMeta
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from sklearndf import ClassifierDF, RegressorDF, SupervisedLearnerDF
from sklearndf.pipeline import SupervisedLearnerPipelineDF

from facet.inspection._explainer import ExplainerFactory, ParallelExplainer
from facet.inspection.shap import (
    ShapCalculator,
    ShapInteractionValuesCalculator,
    ShapValuesCalculator,
)

log = logging.getLogger(__name__)

__all__ = [
    "ClassifierShapCalculator",
    "ClassifierShapInteractionValuesCalculator",
    "ClassifierShapValuesCalculator",
    "LearnerShapCalculator",
    "RegressorShapCalculator",
    "RegressorShapInteractionValuesCalculator",
    "RegressorShapValuesCalculator",
]

#
# Type variables
#

T_ClassifierDF = TypeVar("T_ClassifierDF", bound=ClassifierDF)
T_SupervisedLearnerDF = TypeVar("T_SupervisedLearnerDF", bound=SupervisedLearnerDF)
T_RegressorDF = TypeVar("T_RegressorDF", bound=RegressorDF)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class LearnerShapCalculator(
    ShapCalculator[T_SupervisedLearnerDF],
    Generic[T_SupervisedLearnerDF],
    metaclass=ABCMeta,
):
    """
    Base class for SHAP calculators based on :mod:`sklearndf` learners.
    """

    #: Default name for the feature index (= column index)
    IDX_FEATURE = "feature"

    #: The supervised learner pipeline used to calculate SHAP values.
    pipeline: SupervisedLearnerPipelineDF[T_SupervisedLearnerDF]

    def __init__(
        self,
        pipeline: SupervisedLearnerPipelineDF[T_SupervisedLearnerDF],
        explainer_factory: ExplainerFactory[T_SupervisedLearnerDF],
        *,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        super().__init__(
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.pipeline = pipeline

    def get_feature_names(self) -> pd.Index:
        """[see superclass]"""

        return self.pipeline.final_estimator.feature_names_in_.rename(
            LearnerShapCalculator.IDX_FEATURE
        )

    def _get_shap(self, features: pd.DataFrame) -> pd.DataFrame:

        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self.explainer_factory.uses_background_dataset:
            background_dataset = self.preprocess_features(features)

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

        pipeline = self.pipeline
        explainer = self.explainer_factory.make_explainer(
            model=pipeline.final_estimator, data=background_dataset
        )

        if self.n_jobs != 1:
            explainer = ParallelExplainer(
                explainer,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )

        # we explain all observations using the model, resulting in a matrix of
        # SHAP values for each observation and feature
        return self._calculate_shap(features=features, explainer=explainer)

    def preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """[see superclass]"""

        # get the model
        pipeline = self.pipeline

        # pre-process the features
        if pipeline.preprocessing is not None:
            features = pipeline.preprocessing.transform(features)

        # re-index the features to fit the sequence that was used to fit the learner
        # (in case feature order was shuffled during preprocessing, or train split
        # pre-processing removed columns)
        return features.reindex(
            columns=pipeline.final_estimator.feature_names_in_, copy=False
        )


@inheritdoc(match="""[see superclass]""")
class RegressorShapCalculator(
    LearnerShapCalculator[T_RegressorDF],
    Generic[T_RegressorDF],
    metaclass=ABCMeta,
):
    """
    Calculates SHAP (interaction) values for regression models.
    """

    #: Multi-output SHAP values are determined by target.
    MULTI_OUTPUT_INDEX_NAME = "target"

    def _get_output_names(self, features: pd.DataFrame) -> List[str]:
        regressor_df = self.pipeline.final_estimator
        assert regressor_df is not None, "pipeline must be fitted"

        return regressor_df.output_names_

    def get_multi_output_names(self) -> List[str]:
        """[see superclass]"""
        return self.pipeline.output_names_


class RegressorShapValuesCalculator(
    ShapValuesCalculator[T_RegressorDF],
    RegressorShapCalculator[T_RegressorDF],
    Generic[T_RegressorDF],
):
    """
    Calculates SHAP values for regression models.
    """

    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
            for raw_shap_matrix in raw_shap_tensors
        ]


class RegressorShapInteractionValuesCalculator(
    ShapInteractionValuesCalculator[T_RegressorDF],
    RegressorShapCalculator[T_RegressorDF],
    Generic[T_RegressorDF],
):
    """
    Calculates SHAP interaction matrices for regression models.
    """

    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
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


@inheritdoc(match="""[see superclass]""")
class ClassifierShapCalculator(
    LearnerShapCalculator[T_ClassifierDF],
    Generic[T_ClassifierDF],
    metaclass=ABCMeta,
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

        if n_outputs == 2 and isinstance(shap_tensors, np.ndarray):
            # if we have a single output *and* binary classification, the explainer
            # will have returned a single tensor for the positive class;
            # the SHAP values for the negative class will have the opposite sign
            (shap_tensors,) = super()._convert_shap_tensors_to_list(
                shap_tensors=shap_tensors, n_outputs=1
            )
            return [-shap_tensors, shap_tensors]
        else:
            return super()._convert_shap_tensors_to_list(
                shap_tensors=shap_tensors, n_outputs=n_outputs
            )

    def _get_output_names(self, features: pd.DataFrame) -> Sequence[str]:
        classifier_df = self.pipeline.final_estimator
        assert classifier_df.is_fitted, "classifier must be fitted"

        assert (
            classifier_df.n_outputs_ == 1
        ), "classification model must be single-output"

        try:
            # noinspection PyTypeChecker
            class_names: List[str] = cast(
                npt.NDArray[Any], classifier_df.classes_
            ).tolist()
        except Exception as cause:
            raise AssertionError("classifier must define classes_ attribute") from cause

        n_classes = len(class_names)

        if n_classes == 1:
            raise RuntimeError(
                "cannot explain a (sub)sample with one single category "
                f"{repr(class_names[0])}: "
                "consider using a stratified cross-validation strategy"
            )

        elif n_classes == 2:
            # for binary classifiers, we will generate only output for the first class
            # as the probabilities for the second class are trivially linked to class 1
            return class_names[:1]

        else:
            return class_names

    def get_multi_output_names(self) -> List[str]:
        """[see superclass]"""
        assert (
            self.pipeline.final_estimator.n_outputs_ == 1
        ), "only single-output classifiers are currently supported"
        return list(map(str, self.pipeline.final_estimator.classes_))


class ClassifierShapValuesCalculator(
    ShapValuesCalculator[T_ClassifierDF],
    ClassifierShapCalculator[T_ClassifierDF],
    Generic[T_ClassifierDF],
):
    """
    Calculates SHAP matrices for classification models.
    """

    # noinspection DuplicatedCode
    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        # return a list of data frame [obs x features], one for each of the outputs

        n_arrays = len(raw_shap_tensors)

        if n_arrays == 2:
            # in the binary classification case, we will proceed with SHAP values
            # for class 0 only, since values for class 1 will just be the same
            # values times (*-1)  (the opposite delta probability)

            # to ensure the values are returned as expected above,
            # and no information of class 1 is discarded, assert the
            # following:
            if not np.allclose(raw_shap_tensors[0], -raw_shap_tensors[1]):
                _raw_shap_tensor_totals = raw_shap_tensors[0] + raw_shap_tensors[1]
                log.warning(
                    "shap values of binary classifiers should add up to 0.0 "
                    "for each observation and feature, but total shap values range "
                    f"from {_raw_shap_tensor_totals.min():g} "
                    f"to {_raw_shap_tensor_totals.max():g}"
                )

            # all good: proceed with SHAP values for class 1 (positive class):
            raw_shap_tensors = raw_shap_tensors[1:]

        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
            for raw_shap_matrix in raw_shap_tensors
        ]


class ClassifierShapInteractionValuesCalculator(
    ShapInteractionValuesCalculator[T_ClassifierDF],
    ClassifierShapCalculator[T_ClassifierDF],
    Generic[T_ClassifierDF],
):
    """
    Calculates SHAP interaction matrices for classification models.
    """

    # noinspection DuplicatedCode
    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        # return a list of data frame [(obs x features) x features],
        # one for each of the outputs

        n_arrays = len(raw_shap_tensors)

        if n_arrays == 2:
            # in the binary classification case, we will proceed with SHAP values
            # for class 0 only, since values for class 1 will just be the same
            # values times (*-1)  (the opposite delta probability)

            # to ensure the values are returned as expected above,
            # and no information of class 1 is discarded, assert the
            # following:
            if not np.allclose(raw_shap_tensors[0], -raw_shap_tensors[1]):
                _raw_shap_tensor_totals = raw_shap_tensors[0] + raw_shap_tensors[1]
                log.warning(
                    "shap interaction values of binary classifiers must add up to 0.0 "
                    "for each observation and feature pair, but total shap values "
                    f"range from {_raw_shap_tensor_totals.min():g} "
                    f"to {_raw_shap_tensor_totals.max():g}"
                )

            # all good: proceed with SHAP values for class 1 (positive class):
            raw_shap_tensors = raw_shap_tensors[1:]

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


__tracker.validate()
