"""
Helper classes for SHAP calculations.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin
from pytools.parallelization import Job, JobRunner, ParallelizableMixin
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from ..crossfit import LearnerCrossfit
from ..data import Sample
from ._explainer import BaseExplainer, ExplainerFactory

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

T_Self = TypeVar("T_Self")
T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)

#
# Type definitions
#

ShapToDataFrameFunction = Callable[
    [List[np.ndarray], pd.Index, pd.Index], List[pd.DataFrame]
]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="[see superclass]")
class ShapCalculator(
    FittableMixin[LearnerCrossfit[T_LearnerPipelineDF]],
    ParallelizableMixin,
    Generic[T_LearnerPipelineDF],
    metaclass=ABCMeta,
):
    """
    Base class for all SHAP calculators.

    A SHAP calculator uses the ``shap`` package to calculate SHAP tensors for OOB
    samples across splits of a crossfit, then consolidates and aggregates results
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
        explainer_factory: ExplainerFactory,
        *,
        explain_full_sample: bool,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param explain_full_sample: if ``True``, calculate SHAP values for full sample,
            otherwise only use OOB sample for each crossfit
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.explain_full_sample = explain_full_sample
        self._explainer_factory = explainer_factory
        self.shap_: Optional[pd.DataFrame] = None
        self.feature_index_: Optional[pd.Index] = None
        self.output_names_: Optional[List[str]] = None
        self.sample_: Optional[Sample] = None
        self.n_splits_: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.shap_ is not None

    def fit(
        self: T_Self, crossfit: LearnerCrossfit[T_LearnerPipelineDF], **fit_params
    ) -> T_Self:
        """
        Calculate the SHAP values.

        :param crossfit: the learner crossfit for which to calculate SHAP values
        :param fit_params: additional fit parameters (unused)
        :return: self
        """

        # noinspection PyMethodFirstArgAssignment
        self: ShapCalculator  # support type hinting in PyCharm

        # reset fit in case we get an exception along the way
        self.shap_ = None

        training_sample = crossfit.sample_
        self.feature_index_ = crossfit.pipeline.feature_names_out_.rename(
            Sample.IDX_FEATURE
        )
        self.output_names_ = self._get_output_names(crossfit=crossfit)
        self.sample_ = training_sample

        # calculate shap values and re-order the observation index to match the
        # sequence in the original training sample
        shap_all_splits_df: pd.DataFrame = self._get_shap_all_splits(crossfit=crossfit)

        assert 2 <= shap_all_splits_df.index.nlevels <= 3
        assert shap_all_splits_df.index.names[1] == training_sample.index.name

        self.shap_ = shap_all_splits_df.reindex(
            index=training_sample.index.intersection(
                cast(pd.MultiIndex, shap_all_splits_df.index).levels[1], sort=False
            ),
            level=1,
            copy=False,
        )

        self.n_splits_ = 1 if self.explain_full_sample else crossfit.n_splits_

        return self

    @abstractmethod
    def get_shap_values(self, aggregation: Optional[str]) -> pd.DataFrame:
        """
        The resulting aggregated shap values as a data frame,
        aggregated to averaged SHAP contributions per feature and observation.

        :param aggregation: aggregation method, or ``None`` for no aggregation
        :return: SHAP contribution values with shape
            (n_observations, n_outputs * n_features)
        """

    @abstractmethod
    def get_shap_interaction_values(self, aggregation: Optional[str]) -> pd.DataFrame:
        """
        The resulting aggregated shap interaction values as a data frame,
        aggregated to averaged SHAP interaction values per observation.

        :param aggregation: aggregation method, or ``None`` for no aggregation
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
    def _get_multi_output_names(
        self, model: T_LearnerPipelineDF, sample: Sample
    ) -> List[str]:
        pass

    def _get_shap_all_splits(
        self, crossfit: LearnerCrossfit[T_LearnerPipelineDF]
    ) -> pd.DataFrame:
        crossfit: LearnerCrossfit[LearnerPipelineDF]

        sample = crossfit.sample_

        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self._explainer_factory.uses_background_dataset:
            background_dataset = sample.features
            pipeline = crossfit.pipeline
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

        def _make_explainer(_model: T_LearnerPipelineDF) -> BaseExplainer:
            return self._explainer_factory.make_explainer(
                model=_model.final_estimator,
                # we re-index the columns of the background dataset to match
                # the column sequence of the model (in case feature order
                # was shuffled, or train split pre-processing removed columns)
                data=(
                    None
                    if background_dataset is None
                    else background_dataset.reindex(
                        columns=_model.final_estimator.feature_names_in_,
                        copy=False,
                    )
                ),
            )

        shap_df_per_split: List[pd.DataFrame]

        if self.explain_full_sample:
            # we explain the full sample using the model fitted on the full sample
            # so the result is a list with a single data frame of shap values
            model = crossfit.pipeline
            shap_df_per_split = [
                self._get_shap_for_split(
                    model=model,
                    sample=sample,
                    explainer=_make_explainer(model),
                    features_out=self.feature_index_,
                    shap_matrix_for_split_to_df_fn=self._convert_raw_shap_to_df,
                    multi_output_type=self.get_multi_output_type(),
                    multi_output_names=self._get_multi_output_names(
                        model=model, sample=sample
                    ),
                )
            ]

        else:
            shap_df_per_split = JobRunner.from_parallelizable(self).run_jobs(
                *(
                    Job.delayed(self._get_shap_for_split)(
                        model,
                        sample,
                        _make_explainer(model),
                        self.feature_index_,
                        self._convert_raw_shap_to_df,
                        self.get_multi_output_type(),
                        self._get_multi_output_names(model=model, sample=sample),
                    )
                    for model, sample in zip(
                        crossfit.models(),
                        (
                            sample.subsample(iloc=oob_split)
                            for _, oob_split in crossfit.splits()
                        ),
                    )
                )
            )

        return self._concatenate_splits(shap_df_per_split=shap_df_per_split)

    @abstractmethod
    def _concatenate_splits(
        self, shap_df_per_split: List[pd.DataFrame]
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _aggregate_splits(
        shap_all_splits_df: pd.DataFrame, method: Optional[str]
    ) -> pd.DataFrame:
        # Group SHAP values by observation ID, aggregate SHAP values using mean or std,
        # then restore the original order of observations

        if method is None:
            return shap_all_splits_df

        index = shap_all_splits_df.index
        n_levels = index.nlevels

        assert n_levels > 1
        assert index.names[0] == ShapCalculator.IDX_SPLIT

        level = 1 if n_levels == 2 else tuple(range(1, n_levels))

        if method == ShapCalculator.AGG_MEAN:
            shap_aggregated = shap_all_splits_df.mean(level=level)
        elif method == ShapCalculator.AGG_STD:
            shap_aggregated = shap_all_splits_df.std(level=level)
        else:
            raise ValueError(f"unknown aggregation method: {method}")

        return shap_aggregated

    @staticmethod
    @abstractmethod
    def _get_shap_for_split(
        model: LearnerPipelineDF,
        sample: Sample,
        explainer: BaseExplainer,
        features_out: pd.Index,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _convert_shap_tensors_to_list(
        shap_tensors: Union[np.ndarray, Sequence[np.ndarray]],
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ):
        def _validate_shap_tensor(_t: np.ndarray) -> None:
            if np.isnan(np.sum(_t)):
                raise AssertionError(
                    "Output of SHAP explainer included NaN values. "
                    "This should not happen; consider initialising the "
                    "LearnerInspector with an ExplainerFactory that has a different "
                    "configuration, or that makes SHAP explainers of a different type."
                )

        n_outputs = len(multi_output_names)

        if isinstance(shap_tensors, List):
            for shap_tensor in shap_tensors:
                _validate_shap_tensor(shap_tensor)
        else:
            _validate_shap_tensor(shap_tensors)
            if (
                n_outputs == 2
                and multi_output_type
                == ClassifierShapCalculator.get_multi_output_type()
            ):
                # if we have a single output *and* binary classification, the explainer
                # will have returned a single tensor for the positive class;
                # the SHAP values for the negative class will have the opposite sign
                shap_tensors = [-shap_tensors, shap_tensors]
            else:
                # if we have a single output *and* no classification, the explainer will
                # have returned a single tensor as an array, so we wrap it in a list
                shap_tensors = [shap_tensors]

        if n_outputs != len(shap_tensors):
            raise AssertionError(
                f"count of SHAP tensors (n={len(shap_tensors)}) "
                f"should match number of outputs ({multi_output_names})"
            )

        return shap_tensors

    @staticmethod
    def _preprocess_features(model: LearnerPipelineDF, sample: Sample) -> pd.DataFrame:
        # get the out-of-bag subsample of the training sample, with feature columns
        # in the sequence that was used to fit the learner

        # get the features of all out-of-bag observations
        x = sample.features

        # pre-process the features
        if model.preprocessing is not None:
            x = model.preprocessing.transform(x)

        # re-index the features to fit the sequence that was used to fit the learner
        return x.reindex(columns=model.final_estimator.feature_names_in_, copy=False)

    @staticmethod
    @abstractmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
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

    @staticmethod
    @abstractmethod
    def _get_output_names(crossfit: LearnerCrossfit[T_LearnerPipelineDF]) -> List[str]:
        pass


@inheritdoc(match="[see superclass]")
class ShapValuesCalculator(
    ShapCalculator[T_LearnerPipelineDF], Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP contribution values.
    """

    def get_shap_values(self, aggregation: Optional[str]) -> pd.DataFrame:
        """[see superclass]"""
        self._ensure_fitted()
        return ShapCalculator._aggregate_splits(
            shap_all_splits_df=self.shap_, method=aggregation
        )

    def get_shap_interaction_values(self, aggregation: Optional[str]) -> pd.DataFrame:
        """
        Not implemented.

        :param aggregation: (ignored)
        :return: (never returns)
        :raise TypeError: always raises this - SHAP interaction values are not supported
        """
        raise TypeError(
            f"{type(self).__name__}"
            f".{ShapValuesCalculator.get_shap_interaction_values.__name__}() "
            "is not defined"
        )

    @staticmethod
    def _get_shap_for_split(
        model: LearnerPipelineDF,
        sample: Sample,
        explainer: BaseExplainer,
        features_out: pd.Index,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ) -> pd.DataFrame:
        x = ShapCalculator._preprocess_features(model=model, sample=sample)

        if x.isna().values.any():
            log.warning(
                "preprocessed sample passed to SHAP explainer contains NaN values; "
                "try to change preprocessing to impute all NaN values"
            )

        # calculate the shap values, and ensure the result is a list of arrays
        shap_values: List[np.ndarray] = ShapCalculator._convert_shap_tensors_to_list(
            shap_tensors=explainer.shap_values(x),
            multi_output_type=multi_output_type,
            multi_output_names=multi_output_names,
        )

        # convert to a data frame per output (different logic depending on whether
        # we have a regressor or a classifier, implemented by method
        # shap_matrix_for_split_to_df_fn)
        shap_values_df_per_output: List[pd.DataFrame] = [
            shap.reindex(columns=features_out, copy=False, fill_value=0.0)
            for shap in shap_matrix_for_split_to_df_fn(shap_values, x.index, x.columns)
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

    def get_shap_values(self, aggregation: Optional[str]) -> pd.DataFrame:
        """[see superclass]"""
        self._ensure_fitted()
        return ShapCalculator._aggregate_splits(
            shap_all_splits_df=self.shap_.sum(level=(0, 1)), method=aggregation
        )

    def get_shap_interaction_values(self, aggregation: Optional[str]) -> pd.DataFrame:
        """[see superclass]"""
        self._ensure_fitted()
        return ShapCalculator._aggregate_splits(
            shap_all_splits_df=self.shap_, method=aggregation
        )

    def get_diagonals(self) -> pd.DataFrame:
        """
        The get_diagonals of all SHAP interaction matrices, of shape
        (n_observations, n_outputs * n_features).

        :return: SHAP interaction values with shape
            (n_observations * n_features, n_outputs * n_features), i.e., for each
            observation and output we get the feature interaction values of size
            n_features * n_features.
        """
        self._ensure_fitted()

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

    @staticmethod
    def _get_shap_for_split(
        model: LearnerPipelineDF,
        sample: Sample,
        explainer: BaseExplainer,
        features_out: pd.Index,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ) -> pd.DataFrame:
        x = ShapCalculator._preprocess_features(model=model, sample=sample)

        # calculate the im values (returned as an array)
        try:
            # noinspection PyUnresolvedReferences
            shap_interaction_values_fn = explainer.shap_interaction_values
        except AttributeError:
            raise RuntimeError(
                "Explainer does not implement method shap_interaction_values"
            )

        # calculate the shap interaction values; ensure the result is a list of arrays
        shap_interaction_tensors: List[
            np.ndarray
        ] = ShapCalculator._convert_shap_tensors_to_list(
            shap_tensors=shap_interaction_values_fn(x),
            multi_output_type=multi_output_type,
            multi_output_names=multi_output_names,
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
            for im in shap_matrix_for_split_to_df_fn(
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
class RegressorShapCalculator(ShapCalculator[RegressorPipelineDF], metaclass=ABCMeta):
    """
    Calculates SHAP (interaction) values for regression models.
    """

    @staticmethod
    def _get_output_names(crossfit: LearnerCrossfit[RegressorPipelineDF]) -> List[str]:
        # noinspection PyProtectedMember
        return crossfit.sample_._target_names

    @staticmethod
    def get_multi_output_type() -> str:
        """[see superclass]"""
        return Sample.IDX_TARGET

    def _get_multi_output_names(
        self, model: RegressorPipelineDF, sample: Sample
    ) -> List[str]:
        # noinspection PyProtectedMember
        return sample._target_names

    def _concatenate_splits(
        self, shap_df_per_split: List[pd.DataFrame]
    ) -> pd.DataFrame:
        return pd.concat(
            shap_df_per_split,
            keys=range(len(shap_df_per_split)),
            names=[ShapCalculator.IDX_SPLIT],
        )


class RegressorShapValuesCalculator(
    RegressorShapCalculator, ShapValuesCalculator[RegressorPipelineDF]
):
    """
    Calculates SHAP values for regression models.
    """

    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
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
    RegressorShapCalculator, ShapInteractionValuesCalculator[RegressorPipelineDF]
):
    """
    Calculates SHAP interaction matrices for regression models.
    """

    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
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
class ClassifierShapCalculator(ShapCalculator[ClassifierPipelineDF], metaclass=ABCMeta):
    """
    Calculates SHAP (interaction) values for classification models.
    """

    COL_CLASS = "class"

    @staticmethod
    def _get_output_names(
        crossfit: LearnerCrossfit[ClassifierPipelineDF],
    ) -> Sequence[str]:
        assert not isinstance(
            crossfit.sample_.target_name, list
        ), "classification model is single-output"
        classifier_df = crossfit.pipeline.final_estimator
        assert classifier_df.is_fitted, "classifier used in crossfit must be fitted"

        try:
            # noinspection PyUnresolvedReferences
            output_names = classifier_df.classes_

        except Exception as cause:
            raise AssertionError(
                "classifier used in crossfit must define classes_ attribute"
            ) from cause

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

    def _get_multi_output_names(
        self, model: ClassifierPipelineDF, sample: Sample
    ) -> List[str]:
        assert isinstance(
            sample.target, pd.Series
        ), "only single-output classifiers are currently supported"
        root_classifier = model.final_estimator.native_estimator
        # noinspection PyUnresolvedReferences
        return [str(class_) for class_ in root_classifier.classes_]

    def _concatenate_splits(
        self, shap_df_per_split: List[pd.DataFrame]
    ) -> pd.DataFrame:
        output_names = self.output_names_

        split_keys = range(len(shap_df_per_split))
        if len(output_names) == 1:
            return pd.concat(
                shap_df_per_split, keys=split_keys, names=[ShapCalculator.IDX_SPLIT]
            )

        else:
            # for multi-class classifiers, ensure that all data frames include
            # columns for all classes (even if a class was missing in any split)

            columns = pd.MultiIndex.from_product(
                iterables=[output_names, self.feature_index_],
                names=[self.get_multi_output_type(), self.feature_index_.name],
            )

            return pd.concat(
                [
                    shap_df.reindex(columns=columns, fill_value=0.0)
                    for shap_df in shap_df_per_split
                ],
                keys=split_keys,
                names=[ShapCalculator.IDX_SPLIT],
            )


class ClassifierShapValuesCalculator(
    ClassifierShapCalculator, ShapValuesCalculator[ClassifierPipelineDF]
):
    """
    Calculates SHAP matrices for classification models.
    """

    # noinspection DuplicatedCode
    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
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
            raw_shap_tensors: List[np.ndarray] = raw_shap_tensors[1:]

        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
            for raw_shap_matrix in raw_shap_tensors
        ]


class ClassifierShapInteractionValuesCalculator(
    ClassifierShapCalculator, ShapInteractionValuesCalculator[ClassifierPipelineDF]
):
    """
    Calculates SHAP interaction matrices for classification models.
    """

    # noinspection DuplicatedCode
    @staticmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
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
            raw_shap_tensors: List[np.ndarray] = raw_shap_tensors[1:]

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
