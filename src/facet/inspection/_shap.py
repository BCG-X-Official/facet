"""
Helper classes for SHAP calculations.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin, fitted_only
from pytools.parallelization import ParallelizableMixin
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from ..data import Sample
from ._explainer import BaseExplainer, ExplainerFactory, ParallelExplainer

log = logging.getLogger(__name__)

__all__ = [
    "ShapCalculator",
    "ShapValuesCalculator",
    "ShapInteractionValuesCalculator",
    "RegressorShapCalculator",
    "RegressorShapValuesCalculator",
    "RegressorShapInteractionValuesCalculator",
    "ClassifierShapCalculator",
    "ClassifierShapValuesCalculator",
    "ClassifierShapInteractionValuesCalculator",
]

#
# Type variables
#

T_ShapCalculator = TypeVar("T_ShapCalculator", bound="ShapCalculator[Any]")
T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF[Any])


#
# Constants
#

ASSERTION__CALCULATOR_IS_FITTED = "calculator is fitted"


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="[see superclass]")
class ShapCalculator(
    FittableMixin[Sample],
    ParallelizableMixin,
    Generic[T_LearnerPipelineDF],
    metaclass=ABCMeta,
):
    """
    Base class for all SHAP calculators.

    A SHAP calculator uses the ``shap`` package to calculate SHAP tensors for all
    observations in a given sample, then consolidates and aggregates results
    in a data frame.
    """

    #: constant for "mean" aggregation method, to be passed as arg ``aggregation``
    #: to :class:`.ShapCalculator` methods that implement it
    AGG_MEAN = "mean"

    #: constant for "std" aggregation method, to be passed as arg ``aggregation``
    #: to :class:`.ShapCalculator` methods that implement it
    AGG_STD = "std"

    #: name of index level indicating the split ID
    IDX_SPLIT = "split"

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
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.pipeline = pipeline
        self._explainer_factory = explainer_factory
        self.shap_: Optional[pd.DataFrame] = None
        self.feature_index_: Optional[pd.Index] = None
        self.output_names_: Optional[Sequence[str]] = None
        self.sample_: Optional[Sample] = None

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.shap_ is not None

    def fit(  # type: ignore[override]
        self: T_ShapCalculator,
        sample: Sample,
        **fit_params: Any,
    ) -> T_ShapCalculator:
        """
        Calculate the SHAP values.

        :param sample: the observations for which to calculate SHAP values
        :param fit_params: additional fit parameters (unused)
        :return: self
        """

        # reset fit in case we get an exception along the way
        self.shap_ = None

        self.feature_index_ = self.pipeline.final_estimator.feature_names_in_
        self.output_names_ = self._get_output_names(sample)
        self.sample_ = sample

        # calculate shap values and re-order the observation index to match the
        # sequence in the original training sample
        shap_df: pd.DataFrame = self._get_shap(sample)

        n_levels = shap_df.index.nlevels
        assert 1 <= n_levels <= 2
        assert shap_df.index.names[0] == sample.index.name

        self.shap_ = shap_df.reindex(
            index=sample.index.intersection(
                (
                    shap_df.index
                    if n_levels == 1
                    else cast(pd.MultiIndex, shap_df.index).levels[0]
                ),
                sort=False,
            ),
            level=0,
            copy=False,
        )

        return self

    @abstractmethod
    def get_shap_values(self) -> pd.DataFrame:
        """
        The resulting shap values, per observation and feature, as a data frame.

        :return: SHAP contribution values with shape
            (n_observations, n_outputs * n_features)
        """

    @abstractmethod
    def get_shap_interaction_values(self) -> pd.DataFrame:
        """
        Get the resulting shap interaction values as a data frame.

        :return: SHAP contribution values with shape
            (n_observations * n_features, n_outputs * n_features)
        :raise TypeError: this SHAP calculator does not support interaction values
        """

    @staticmethod
    @abstractmethod
    def get_multi_output_type() -> str:
        """
        :return: a category name for the dimensions represented by multiple outputs
        """

    @abstractmethod
    def get_multi_output_names(self, sample: Sample) -> List[str]:
        """
        :return: a name for each of the outputs
        """
        pass

    def _get_shap(self, sample: Sample) -> pd.DataFrame:

        pipeline = self.pipeline

        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self._explainer_factory.uses_background_dataset:
            background_dataset = sample.features
            if pipeline.preprocessing:
                background_dataset = pipeline.preprocessing.transform(
                    X=background_dataset
                )

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

        explainer = self._explainer_factory.make_explainer(
            model=pipeline.final_estimator,
            # we re-index the columns of the background dataset to match
            # the column sequence of the model (in case feature order
            # was shuffled, or train split pre-processing removed columns)
            data=(
                None
                if background_dataset is None
                else background_dataset.reindex(
                    columns=pipeline.final_estimator.feature_names_in_,
                    copy=False,
                )
            ),
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

    @abstractmethod
    def _calculate_shap(
        self, *, sample: Sample, explainer: BaseExplainer
    ) -> pd.DataFrame:
        pass

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

    def _preprocess_features(self, sample: Sample) -> pd.DataFrame:

        # get the model
        pipeline = self.pipeline

        # get the features of all out-of-bag observations
        x = sample.features

        # pre-process the features
        if pipeline.preprocessing is not None:
            x = pipeline.preprocessing.transform(x)

        # re-index the features to fit the sequence that was used to fit the learner
        return x.reindex(columns=pipeline.final_estimator.feature_names_in_, copy=False)

    @staticmethod
    @abstractmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        """
        Convert the SHAP tensors for a single split to a data frame.

        :param raw_shap_tensors: the raw values returned by the SHAP explainer
        :param observations: the ids used for indexing the explained observations
        :param features_in_split: the features in the current split,
            explained by the SHAP explainer
        :return: SHAP values of a single split as data frame
        """
        pass

    @abstractmethod
    def _get_output_names(self, sample: Sample) -> Sequence[str]:
        pass


@inheritdoc(match="[see superclass]")
class ShapValuesCalculator(
    ShapCalculator[T_LearnerPipelineDF], Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP contribution values.
    """

    @fitted_only
    def get_shap_values(self) -> pd.DataFrame:
        """[see superclass]"""
        return self.shap_

    def get_shap_interaction_values(self) -> pd.DataFrame:
        """
        Not implemented.

        :return: (never returns anything)
        :raise TypeError: SHAP interaction values are not supported - always raised
        """
        raise TypeError(
            f"{type(self).__name__}"
            f".{ShapValuesCalculator.get_shap_interaction_values.__name__}() "
            "is not defined"
        )

    def _calculate_shap(
        self, *, sample: Sample, explainer: BaseExplainer
    ) -> pd.DataFrame:
        x = self._preprocess_features(sample=sample)

        if x.isna().values.any():
            log.warning(
                "preprocessed sample passed to SHAP explainer contains NaN values; "
                "try to change preprocessing to impute all NaN values"
            )

        multi_output_type = self.get_multi_output_type()
        multi_output_names = self.get_multi_output_names(sample=sample)
        assert self.feature_index_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        features_out = self.feature_index_

        # calculate the shap values, and ensure the result is a list of arrays
        shap_values: List[npt.NDArray[np.float_]] = self._convert_shap_tensors_to_list(
            shap_tensors=explainer.shap_values(x), n_outputs=len(multi_output_names)
        )

        # convert to a data frame per output (different logic depending on whether
        # we have a regressor or a classifier, implemented by method
        # shap_matrix_for_split_to_df_fn)
        shap_values_df_per_output: List[pd.DataFrame] = [
            shap.reindex(columns=features_out, copy=False, fill_value=0.0)
            for shap in self._convert_raw_shap_to_df(shap_values, x.index, x.columns)
        ]

        # if we have a single output, return the data frame for that output;
        # else, add a top level to the column index indicating each output

        if len(shap_values_df_per_output) == 1:
            return shap_values_df_per_output[0]
        else:
            return pd.concat(
                shap_values_df_per_output,
                axis=1,
                keys=multi_output_names,
                names=[multi_output_type, features_out.name],
            )


@inheritdoc(match="[see superclass]")
class ShapInteractionValuesCalculator(
    ShapCalculator[T_LearnerPipelineDF], Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP interaction values.
    """

    @fitted_only
    def get_shap_values(self) -> pd.DataFrame:
        """[see superclass]"""

        assert self.shap_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        return self.shap_.groupby(level=0).sum()

    @fitted_only
    def get_shap_interaction_values(self) -> pd.DataFrame:
        """[see superclass]"""

        assert self.shap_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        return self.shap_

    @fitted_only
    def get_diagonals(self) -> pd.DataFrame:
        """
        The get_diagonals of all SHAP interaction matrices, of shape
        (n_observations, n_outputs * n_features).

        :return: SHAP interaction values with shape
            (n_observations * n_features, n_outputs * n_features), i.e., for each
            observation and output we get the feature interaction values of size
            n_features * n_features.
        """

        assert (
            self.shap_ is not None
            and self.sample_ is not None
            and self.feature_index_ is not None
        ), ASSERTION__CALCULATOR_IS_FITTED

        n_observations = len(self.sample_)
        n_features = len(self.feature_index_)
        interaction_matrix = self.shap_

        return pd.DataFrame(
            np.diagonal(
                interaction_matrix.values.reshape(
                    (n_observations, n_features, -1, n_features)
                    # observations x features x outputs x features
                ),
                axis1=1,
                axis2=3,
            ).reshape((n_observations, -1)),
            # observations x (outputs * features)
            index=cast(pd.MultiIndex, interaction_matrix.index).levels[0],
            columns=interaction_matrix.columns,
        )

    def _calculate_shap(
        self, *, sample: Sample, explainer: BaseExplainer
    ) -> pd.DataFrame:
        x = self._preprocess_features(sample=sample)

        multi_output_type = self.get_multi_output_type()
        multi_output_names = self.get_multi_output_names(sample)
        assert self.feature_index_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        features_out = self.feature_index_

        # calculate the shap interaction values; ensure the result is a list of arrays
        shap_interaction_tensors: List[
            npt.NDArray[np.float_]
        ] = self._convert_shap_tensors_to_list(
            shap_tensors=explainer.shap_interaction_values(x),
            n_outputs=len(multi_output_names),
        )

        interaction_matrix_per_output: List[pd.DataFrame] = [
            im.reindex(
                index=pd.MultiIndex.from_product(
                    iterables=(x.index, features_out),
                    names=(x.index.name, features_out.name),
                ),
                columns=features_out,
                copy=False,
                fill_value=0.0,
            )
            for im in self._convert_raw_shap_to_df(
                shap_interaction_tensors, x.index, x.columns
            )
        ]

        # if we have a single output, use the data frame for that output;
        # else, concatenate the values data frame for all outputs horizontally
        # and add a top level to the column index indicating each output
        if len(interaction_matrix_per_output) == 1:
            return interaction_matrix_per_output[0]
        else:
            return pd.concat(
                interaction_matrix_per_output,
                axis=1,
                keys=multi_output_names,
                names=[multi_output_type, features_out.name],
            )


@inheritdoc(match="[see superclass]")
class RegressorShapCalculator(
    ShapCalculator[RegressorPipelineDF[Any]], metaclass=ABCMeta
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


class RegressorShapValuesCalculator(
    RegressorShapCalculator, ShapValuesCalculator[RegressorPipelineDF[Any]]
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
    RegressorShapCalculator, ShapInteractionValuesCalculator[RegressorPipelineDF[Any]]
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


@inheritdoc(match="[see superclass]")
class ClassifierShapCalculator(
    ShapCalculator[ClassifierPipelineDF[Any]], metaclass=ABCMeta
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


class ClassifierShapValuesCalculator(
    ClassifierShapCalculator, ShapValuesCalculator[ClassifierPipelineDF[Any]]
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
    ClassifierShapCalculator, ShapInteractionValuesCalculator[ClassifierPipelineDF[Any]]
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
