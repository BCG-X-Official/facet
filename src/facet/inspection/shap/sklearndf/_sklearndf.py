"""
Implementation of package ``facet.inspection.shap.learner``.
"""

import logging
from abc import ABCMeta
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from facet.data import Sample
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

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF[Any])


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class LearnerShapCalculator(
    ShapCalculator, Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for SHAP calculators based on :mod:`sklearndf` learners.
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        explainer_factory: ExplainerFactory,
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
        return self.pipeline.feature_names_out_.rename(Sample.IDX_FEATURE)

    def _get_shap(self, sample: Sample) -> pd.DataFrame:

        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self._explainer_factory.uses_background_dataset:
            background_dataset = self.preprocess_features(sample)

            background_dataset_not_na = background_dataset.dropna()

            if len(background_dataset_not_na) != len(background_dataset):
                n_original = len(background_dataset)
                n_dropped = n_original - len(background_dataset_not_na)
                log.warning(
                    f"{n_dropped} out of {n_original} observations in the sample "
                    "contain NaN values after pre-processing and will not be included "
                    "in the background dataset"
                )

                background_dataset = background_dataset_not_na

        else:
            background_dataset = None

        pipeline = self.pipeline
        explainer = self._explainer_factory.make_explainer(
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

        # we explain the full sample using the model fitted on the full sample
        # so the result is a list with a single data frame of shap values
        return self._calculate_shap(sample=sample, explainer=explainer)

    def _convert_shap_tensors_to_list(
        self,
        *,
        shap_tensors: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]],
        n_outputs: int,
    ) -> List[npt.NDArray[np.float_]]:
        def _validate_shap_tensor(_t: npt.NDArray[np.float_]) -> None:
            if np.isnan(np.sum(_t)):
                raise AssertionError(
                    "Output of SHAP explainer includes NaN values. "
                    "This should not happen; consider initialising the "
                    "LearnerInspector with an ExplainerFactory that has a different "
                    "configuration, or that makes SHAP explainers of a different type."
                )

        if isinstance(shap_tensors, List):
            for shap_tensor in shap_tensors:
                _validate_shap_tensor(shap_tensor)
        else:
            _validate_shap_tensor(shap_tensors)
            shap_tensors = [shap_tensors]

        if n_outputs != len(shap_tensors):
            raise AssertionError(
                f"count of SHAP tensors (n={len(shap_tensors)}) "
                f"should match number of outputs (n={n_outputs})"
            )

        return shap_tensors

    def preprocess_features(self, sample: Sample) -> pd.DataFrame:

        # get the model
        pipeline = self.pipeline

        # get the features of all out-of-bag observations
        x = sample.features

        # pre-process the features
        if pipeline.preprocessing is not None:
            x = pipeline.preprocessing.transform(x)

        # re-index the features to fit the sequence that was used to fit the learner
        # (in case feature order was shuffled during preprocessing, or train split
        # pre-processing removed columns)
        return x.reindex(columns=pipeline.final_estimator.feature_names_in_, copy=False)


@inheritdoc(match="""[see superclass]""")
class RegressorShapCalculator(
    LearnerShapCalculator[RegressorPipelineDF[Any]], metaclass=ABCMeta
):
    """
    Calculates SHAP (interaction) values for regression models.
    """

    def _get_output_names(self, sample: Sample) -> List[str]:
        # noinspection PyProtectedMember
        return sample._target_names

    @staticmethod
    def get_multi_output_type() -> str:
        """[see superclass]"""
        return Sample.IDX_TARGET

    def get_multi_output_names(self, sample: Sample) -> List[str]:
        """[see superclass]"""
        # noinspection PyProtectedMember
        return sample._target_names


class RegressorShapValuesCalculator(ShapValuesCalculator, RegressorShapCalculator):
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
    ShapInteractionValuesCalculator, RegressorShapCalculator
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
    LearnerShapCalculator[ClassifierPipelineDF[Any]], metaclass=ABCMeta
):
    """
    Calculates SHAP (interaction) values for classification models.
    """

    COL_CLASS = "class"

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

    def _get_output_names(
        self,
        sample: Sample,
    ) -> Sequence[str]:
        assert not isinstance(
            sample.target_name, list
        ), "classification model is single-output"
        classifier_df = self.pipeline.final_estimator
        assert classifier_df.is_fitted, "classifier must be fitted"

        try:
            # noinspection PyTypeChecker
            output_names: List[str] = classifier_df.classes_.tolist()

        except Exception as cause:
            raise AssertionError("classifier must define classes_ attribute") from cause

        n_outputs = len(output_names)

        if n_outputs == 1:
            raise RuntimeError(
                "cannot explain a (sub)sample with one single category "
                f"{repr(output_names[0])}: "
                "consider using a stratified cross-validation strategy"
            )

        elif n_outputs == 2:
            # for binary classifiers, we will generate only output for the first class
            # as the probabilities for the second class are trivially linked to class 1
            return output_names[:1]

        else:
            return output_names

    @staticmethod
    def get_multi_output_type() -> str:
        """[see superclass]"""
        return ClassifierShapCalculator.COL_CLASS

    def get_multi_output_names(self, sample: Sample) -> List[str]:
        """[see superclass]"""
        assert isinstance(
            sample.target, pd.Series
        ), "only single-output classifiers are currently supported"
        root_classifier = self.pipeline.final_estimator.native_estimator
        # noinspection PyUnresolvedReferences
        return list(map(str, root_classifier.classes_))


class ClassifierShapValuesCalculator(ShapValuesCalculator, ClassifierShapCalculator):
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
    ShapInteractionValuesCalculator, ClassifierShapCalculator
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
