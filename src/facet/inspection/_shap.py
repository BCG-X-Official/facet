"""
Helper classes for SHAP calculations
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Generic, List, Optional, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
from shap.explainers.explainer import Explainer

from pytools.api import AllTracker
from pytools.fit import FittableMixin
from pytools.parallelization import ParallelizableMixin
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from ._explainer import ExplainerFactory
from facet import Sample
from facet.crossfit import LearnerCrossfit

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

T = TypeVar("T")
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


class ShapCalculator(
    FittableMixin[LearnerCrossfit[T_LearnerPipelineDF]],
    ParallelizableMixin,
    Generic[T_LearnerPipelineDF],
    metaclass=ABCMeta,
):
    """
    Base class for all SHAP calculators.

    A SHAP calculator uses the ``shap`` package to calculate SHAP tensors for oob
    samples across splits of a crossfit, then consolidates and aggregates results
    in a data frame.
    """

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
            otherwise only use oob sample for each crossfit
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

    def fit(self: T, crossfit: LearnerCrossfit[T_LearnerPipelineDF], **fit_params) -> T:
        """
        Calculate the SHAP values.

        :return: self
        """

        # noinspection PyMethodFirstArgAssignment
        self: ShapCalculator  # support type hinting in PyCharm

        # reset fit in case we get an exception along the way
        self.shap_ = None

        training_sample = crossfit.sample
        self.feature_index_ = crossfit.pipeline.features_out_.rename(Sample.IDX_FEATURE)
        self.output_names_ = self._output_names(crossfit=crossfit)
        self.sample_ = training_sample

        # calculate shap values and re-order the observation index to match the
        # sequence in the original training sample
        shap_all_splits_df: pd.DataFrame = self._shap_all_splits(crossfit=crossfit)

        assert shap_all_splits_df.index.nlevels > 1
        assert shap_all_splits_df.index.names[1] == training_sample.index.name

        self.shap_ = shap_all_splits_df.reindex(
            index=training_sample.index.intersection(
                shap_all_splits_df.index.levels[1], sort=False
            ),
            level=1,
            copy=False,
        )

        return self

    # noinspection PyMissingOrEmptyDocstring
    @property
    def is_fitted(self) -> bool:
        return self.shap_ is not None

    is_fitted.__doc__ = FittableMixin.is_fitted.__doc__

    @abstractmethod
    def get_shap_values(self, consolidate: Optional[str] = None) -> pd.DataFrame:
        """
        The resulting consolidated shap values as a data frame,
        aggregated to averaged SHAP contributions per feature and observation.

        :param consolidate: consolidation method, or ``None`` for no consolidation
        :return: SHAP contribution values with shape \
            (n_observations, n_outputs * n_features).
        """
        pass

    @property
    @abstractmethod
    def shap_columns(self) -> pd.Index:
        """
        The column index of the data frame returned by :meth:`.shap_values`
        """
        pass

    @staticmethod
    @abstractmethod
    def multi_output_type() -> str:
        pass

    @abstractmethod
    def _multi_output_names(
        self, model: T_LearnerPipelineDF, sample: Sample
    ) -> List[str]:
        pass

    def _shap_all_splits(
        self, crossfit: LearnerCrossfit[T_LearnerPipelineDF]
    ) -> pd.DataFrame:
        crossfit: LearnerCrossfit[LearnerPipelineDF]

        sample = crossfit.sample

        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self._explainer_factory.uses_background_dataset:
            background_dataset = sample.features
            pipeline = crossfit.pipeline
            if pipeline.preprocessing:
                background_dataset = pipeline.preprocessing.transform(
                    X=background_dataset
                )

            background_dataset_notna = background_dataset.dropna()

            if len(background_dataset_notna) != len(background_dataset):
                n_original = len(background_dataset)
                n_dropped = n_original - len(background_dataset_notna)
                log.warning(
                    f"{n_dropped} out of {n_original} observations in the sample "
                    "contain NaN values after pre-processing and will not be included "
                    "in the background dataset"
                )

                background_dataset = background_dataset_notna

        else:
            background_dataset = None

        with self._parallel() as parallel:
            shap_df_per_split: List[pd.DataFrame] = parallel(
                self._delayed(self._shap_for_split)(
                    model,
                    sample,
                    self._explainer_factory.make_explainer(
                        model=model.final_estimator,
                        # we re-index the columns of the background dataset to match
                        # the column sequence of the model (in case feature order
                        # was shuffled, or train split pre-processing removed columns)
                        data=(
                            None
                            if background_dataset is None
                            else background_dataset.reindex(
                                columns=model.final_estimator.features_in_, copy=False
                            )
                        ),
                    ),
                    self.feature_index_,
                    self._raw_shap_to_df,
                    self.multi_output_type(),
                    self._multi_output_names(model=model, sample=sample),
                )
                for model, sample in zip(
                    crossfit.models(),
                    (
                        # if we explain full samples, we get samples from an
                        # infinite iterator of the full training sample
                        iter(lambda: sample, None)
                        if self.explain_full_sample
                        # otherwise we iterate over the test splits of each crossfit
                        else (
                            sample.subsample(iloc=oob_split)
                            for _, oob_split in crossfit.splits()
                        )
                    ),
                )
            )

        return self._concatenate_splits(shap_df_per_split=shap_df_per_split)

    @abstractmethod
    def _concatenate_splits(
        self, shap_df_per_split: List[pd.DataFrame]
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _consolidate_splits(
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

        if method == "mean":
            shap_consolidated = shap_all_splits_df.mean(level=level)
        elif method == "std":
            shap_consolidated = shap_all_splits_df.std(level=level)
        else:
            raise ValueError(f"unknown consolidation method: {method}")

        return shap_consolidated

    @staticmethod
    @abstractmethod
    def _shap_for_split(
        model: LearnerPipelineDF,
        sample: Sample,
        explainer: Explainer,
        features_out: pd.Index,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _shap_tensors_to_list(
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
                and multi_output_type == ClassifierShapCalculator.multi_output_type()
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
    def _preprocessed_features(
        model: LearnerPipelineDF, sample: Sample
    ) -> pd.DataFrame:
        # get the out-of-bag subsample of the training sample, with feature columns
        # in the sequence that was used to fit the learner

        # get the features of all out-of-bag observations
        x = sample.features

        # pre-process the features
        if model.preprocessing is not None:
            x = model.preprocessing.transform(x)

        # re-index the features to fit the sequence that was used to fit the learner
        return x.reindex(columns=model.final_estimator.features_in_, copy=False)

    @staticmethod
    @abstractmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        """
        Convert the SHAP tensors for a single split to a data frame.

        :param raw_shap_tensors: the raw values returned by the SHAP explainer
        :param observations: the ids used for indexing the explained observations
        :param features_in_split: the features in the current split, \
            explained by the SHAP explainer
        :return: SHAP values of a single split as data frame
        """
        pass

    @staticmethod
    @abstractmethod
    def _output_names(crossfit: LearnerCrossfit[T_LearnerPipelineDF]) -> List[str]:
        pass


class ShapValuesCalculator(
    ShapCalculator[T_LearnerPipelineDF], Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP contribution values.
    """

    # noinspection PyMissingOrEmptyDocstring
    def get_shap_values(self, consolidate: Optional[str] = None) -> pd.DataFrame:
        self._ensure_fitted()
        return ShapCalculator._consolidate_splits(
            shap_all_splits_df=self.shap_, method=consolidate
        )

    get_shap_values.__doc__ = ShapCalculator.get_shap_values.__doc__

    # noinspection PyMissingOrEmptyDocstring
    @property
    def shap_columns(self) -> pd.Index:
        return self.shap_.columns

    shap_columns.__doc__ = ShapCalculator.shap_columns.__doc__

    @staticmethod
    def _shap_for_split(
        model: LearnerPipelineDF,
        sample: Sample,
        explainer: Explainer,
        features_out: pd.Index,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ) -> pd.DataFrame:
        x = ShapCalculator._preprocessed_features(model=model, sample=sample)

        if x.isna().values.any():
            log.warning(
                "preprocessed sample passed to SHAP explainer contains NaN values; "
                "try to change preprocessing to impute all NaN values"
            )

        # calculate the shap values, and ensure the result is a list of arrays
        shap_values: List[np.ndarray] = ShapCalculator._shap_tensors_to_list(
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


class ShapInteractionValuesCalculator(
    ShapCalculator[T_LearnerPipelineDF], Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP interaction values.
    """

    # noinspection PyMissingOrEmptyDocstring
    def get_shap_values(self, consolidate: Optional[str] = None) -> pd.DataFrame:
        self._ensure_fitted()
        return ShapCalculator._consolidate_splits(
            shap_all_splits_df=self.shap_.sum(level=(0, 1)), method=consolidate
        )

    get_shap_values.__doc__ = ShapCalculator.get_shap_values.__doc__

    def get_shap_interaction_values(
        self, consolidate: Optional[str] = None
    ) -> pd.DataFrame:
        """
        The resulting consolidated shap interaction values as a data frame,
        aggregated to averaged SHAP interaction values per observation.
        """
        self._ensure_fitted()
        return ShapCalculator._consolidate_splits(
            shap_all_splits_df=self.shap_, method=consolidate
        )

    @property
    def shap_columns(self) -> pd.Index:
        """
        The column index of the data frame returned by :meth:`.shap_values`
        and :meth:`.shap_interaction_values`
        """
        return self.shap_.columns

    def diagonals(self) -> pd.DataFrame:
        """
        The diagonals of all SHAP interaction matrices, of shape
        (n_observations, n_outputs * n_features)

        :return: SHAP interaction values with shape \
            (n_observations * n_features, n_outputs * n_features), i.e., for each \
            observation and output we get the feature interaction values of size \
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
            index=interaction_matrix.index.levels[0],
            columns=interaction_matrix.columns,
        )

    @staticmethod
    def _shap_for_split(
        model: LearnerPipelineDF,
        sample: Sample,
        explainer: Explainer,
        features_out: pd.Index,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
        multi_output_type: str,
        multi_output_names: Sequence[str],
    ) -> pd.DataFrame:
        x = ShapCalculator._preprocessed_features(model=model, sample=sample)

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
        ] = ShapCalculator._shap_tensors_to_list(
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


class RegressorShapCalculator(ShapCalculator[RegressorPipelineDF], metaclass=ABCMeta):
    """
    Calculates SHAP (interaction) values for regression models.
    """

    @staticmethod
    def _output_names(crossfit: LearnerCrossfit[RegressorPipelineDF]) -> List[str]:
        return crossfit.sample.target_columns

    @staticmethod
    def multi_output_type() -> str:
        return Sample.IDX_TARGET

    def _multi_output_names(
        self, model: RegressorPipelineDF, sample: Sample
    ) -> List[str]:
        return sample.target_columns

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
    def _raw_shap_to_df(
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
    def _raw_shap_to_df(
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


class ClassifierShapCalculator(ShapCalculator[ClassifierPipelineDF], metaclass=ABCMeta):
    """
    Calculates SHAP (interaction) values for classification models.
    """

    COL_CLASS = "class"

    @staticmethod
    def _output_names(crossfit: LearnerCrossfit[ClassifierPipelineDF]) -> List[str]:
        assert (
            len(crossfit.sample.target_columns) == 1
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
    def multi_output_type() -> str:
        return ClassifierShapCalculator.COL_CLASS

    def _multi_output_names(
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

        index_names = [ShapCalculator.IDX_SPLIT, *shap_df_per_split[0].index.names]

        split_keys = range(len(shap_df_per_split))
        if len(output_names) == 1:
            return pd.concat(shap_df_per_split, keys=split_keys, names=index_names)

        else:
            # for multi-class classifiers, ensure that all data frames include
            # columns for all classes (even if a class was missing in any split)

            columns = pd.MultiIndex.from_product(
                iterables=[output_names, self.feature_index_],
                names=[self.multi_output_type(), self.feature_index_.name],
            )

            return pd.concat(
                [
                    shap_df.reindex(columns=columns, fill_value=0.0)
                    for shap_df in shap_df_per_split
                ],
                keys=split_keys,
                names=index_names,
            )


class ClassifierShapValuesCalculator(
    ClassifierShapCalculator, ShapValuesCalculator[ClassifierPipelineDF]
):
    """
    Calculates SHAP matrices for classification models.
    """

    @staticmethod
    def _raw_shap_to_df(
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

            # all good: proceed with SHAP values for class 0:
            raw_shap_tensors = raw_shap_tensors[:1]

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

    @staticmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        # return a list of data frame [(obs x features) x features],
        # one for each of the outputs

        n_arrays = len(raw_shap_tensors)

        if n_arrays == 2:
            # in the binary classification case, we will proceed with SHAP values
            # for class 0, since values for class 1 will just be the same
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

            # all good: proceed with SHAP values for class 0:
            raw_shap_tensors = raw_shap_tensors[:1]

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
