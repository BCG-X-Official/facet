"""
Helper classes for SHAP calculations
"""

import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from gamma.common.fit import FittableMixin
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.sklearndf.pipeline import BaseLearnerPipelineDF

log = logging.getLogger(__name__)

#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_Self = TypeVar("T_Self")

#
# Type definitions
#

ExplainerFactory = Callable[[BaseEstimator, pd.DataFrame], Explainer]

ShapToDataFrameFunction = Callable[
    [List[np.ndarray], np.ndarray, pd.Index], List[pd.DataFrame]
]


#
# Class definitions
#


class BaseShapCalculator(
    FittableMixin[LearnerCrossfit[T_LearnerPipelineDF]],
    ParallelizableMixin,
    ABC,
    Generic[T_LearnerPipelineDF],
):
    """
    Base class for all SHAP calculators.

    A SHAP calculator uses the `shap` package to calculate SHAP tensors for oob
    samples across splits of a crossfit, then consolidates and aggregates results
    in a data frame.
    """

    def __init__(
        self,
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
        self._explainer_factory = explainer_factory
        self.shap_: Optional[pd.DataFrame] = None
        self.feature_index_: Optional[pd.Index] = None
        self.target_columns_: Optional[List[str]] = None
        self.n_observations_: Optional[int] = None

    def fit(
        self: T_Self, crossfit: LearnerCrossfit[T_LearnerPipelineDF], **fit_params
    ) -> T_Self:
        """
        Calculate the SHAP values.

        :return: self
        """

        # reset fit in case we get an exception along the way
        self.shap_ = None

        training_sample = crossfit.training_sample
        self.feature_index_ = crossfit.pipeline.features_out.rename(Sample.COL_FEATURE)
        self.target_columns_ = training_sample.target_columns
        self.n_observations_ = len(training_sample)

        # calculate shap values and re-order the observation index to match the
        # sequence in the original training sample
        self.shap_ = self._consolidate_splits(
            self._shap_all_splits(crossfit=crossfit),
            observation_index=training_sample.index,
        )

        return self

    # noinspection PyMissingOrEmptyDocstring
    @property
    def is_fitted(self) -> bool:
        return self.shap_ is not None

    is_fitted.__doc__ = FittableMixin.is_fitted.__doc__

    @property
    def shap_values(self) -> pd.DataFrame:
        """
        The resulting consolidated as a data frame, aggregated to averaged SHAP
        values per observation.

        The format of the data frame varies depending on the nature of the SHAP
        calculation, see documentation for implementations of this base class.
        """
        self._ensure_fitted()
        return self.shap_

    def _shap_all_splits(
        self, crossfit: LearnerCrossfit[T_LearnerPipelineDF]
    ) -> pd.DataFrame:
        explainer_factory = self._explainer_factory
        training_sample = crossfit.training_sample

        with self._parallel() as parallel:
            shap_df_per_split = parallel(
                self._delayed(self._shap_for_split)(
                    model,
                    training_sample,
                    oob_split,
                    self.feature_index_,
                    explainer_factory,
                    self._raw_shap_to_df,
                )
                for model, (_train_split, oob_split) in zip(
                    crossfit.models(), crossfit.splits()
                )
            )
        return pd.concat(shap_df_per_split)

    @abstractmethod
    def _consolidate_splits(
        self, shap_all_splits_df: pd.DataFrame, observation_index: pd.Index
    ) -> pd.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def _shap_for_split(
        model: BaseLearnerPipelineDF,
        training_sample: Sample,
        oob_split: np.ndarray,
        features_out: pd.Index,
        explainer_factory_fn: ExplainerFactory,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
    ) -> pd.DataFrame:
        pass

    @staticmethod
    def _x_oob(
        model: BaseLearnerPipelineDF, training_sample: Sample, oob_split: np.ndarray
    ) -> pd.DataFrame:
        # get the out-of-bag subsample of the training sample, with feature columns
        # in the sequence that was used to fit the learner

        # get the features of all out-of-bag observations
        x_oob = training_sample.subsample(loc=oob_split).features

        # pre-process the features
        if model.preprocessing is not None:
            x_oob = model.preprocessing.transform(x_oob)

        # re-index the features to fit the sequence that was used to fit the learner
        x_oob = x_oob.reindex(columns=model.final_estimator.features_in, copy=False)

        return x_oob

    @staticmethod
    @abstractmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: np.ndarray,
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
    def _make_column_index(targets: Optional[pd.Index], features: pd.Index):
        # make a (multi) index from an optional target index and a feature index
        if targets is None:
            return features
        else:
            return pd.MultiIndex.from_product((targets, features))


class ShapValuesCalculator(
    BaseShapCalculator[T_LearnerPipelineDF], ABC, Generic[T_LearnerPipelineDF]
):
    """
    Base class for SHAP values calculations.

    The :attr:`.shap_values` property returns SHAP values with shape
    (n_observations, n_targets * n_features).
    """

    def _consolidate_splits(
        self, shap_all_splits_df: pd.DataFrame, observation_index: pd.Index
    ) -> pd.DataFrame:
        # Group SHAP values by observation ID, aggregate SHAP values using mean(),
        # then restore the original order of observations
        mean_per_split = shap_all_splits_df.groupby(
            level=0, sort=False, observed=True
        ).mean()
        return mean_per_split.reindex(
            index=observation_index.intersection(mean_per_split.index, sort=False),
            copy=False,
        )

    @staticmethod
    def _shap_for_split(
        model: BaseLearnerPipelineDF,
        training_sample: Sample,
        oob_split: np.ndarray,
        features_out: pd.Index,
        explainer_factory_fn: ExplainerFactory,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
    ) -> pd.DataFrame:
        x_oob = BaseShapCalculator._x_oob(model, training_sample, oob_split)

        # calculate the shap values (returned as an array)
        shap_values: np.ndarray = explainer_factory_fn(
            model.final_estimator.root_estimator, x_oob
        ).shap_values(x_oob)

        if isinstance(shap_values, np.ndarray):
            # if we have a single target *and* no classification, the explainer will
            # have returned a single tensor as an array
            shap_values: List[np.ndarray] = [shap_values]

        # convert to a data frame per target (different logic depending on whether
        # we have a regressor or a classifier, implemented by method
        # shap_matrix_for_split_to_df_fn)
        shap_values_df_per_target: List[pd.DataFrame] = [
            shap.reindex(columns=features_out, copy=False, fill_value=0.0)
            for shap in shap_matrix_for_split_to_df_fn(
                shap_values, oob_split, x_oob.columns
            )
        ]

        # if we have a single target, return the data frame for that target;
        # else, add a top level to the column index indicating each target

        if len(shap_values_df_per_target) == 1:
            return shap_values_df_per_target[0]
        else:
            assert training_sample.n_targets > 1
            return pd.concat(
                shap_values_df_per_target,
                axis=1,
                keys=training_sample.target_columns,
                names=[Sample.COL_TARGET],
            )


class ShapInteractionValuesCalculator(
    BaseShapCalculator[T_LearnerPipelineDF], ABC, Generic[T_LearnerPipelineDF]
):
    """
    Base class for SHAP interaction values calculations.

    The :attr:`.shap_values` property returns SHAP values with shape
    (n_observations * n_features, n_targets * n_features), i.e., for each observation
    and target we get the feature interaction values of size n_features * n_features.
    """

    def diagonals(self) -> pd.DataFrame:
        """
        The diagonals of all SHAP interaction matrices, of shape
        (n_observations, n_targets * n_features)
        """
        self._ensure_fitted()

        n_observations = self.n_observations_
        n_features = len(self.feature_index_)
        n_targets = len(self.target_columns_)
        interaction_matrix = self.shap_

        return pd.DataFrame(
            np.diagonal(
                interaction_matrix.values.reshape(
                    (n_observations, n_features, n_targets, n_features)
                ),
                axis1=1,
                axis2=3,
            ).reshape((n_observations, n_targets * n_features)),
            index=interaction_matrix.index.levels[0],
            columns=interaction_matrix.columns,
        )

    def _consolidate_splits(
        self, shap_all_splits_df: pd.DataFrame, observation_index: pd.Index
    ) -> pd.DataFrame:
        # Group SHAP values by observation ID and feature, and aggregate using mean()
        # return shap_all_splits_df
        return (
            shap_all_splits_df.groupby(level=(0, 1), sort=False, observed=True)
            .mean()
            .reindex(index=observation_index, level=0)
        )

    @staticmethod
    def _shap_for_split(
        model: BaseLearnerPipelineDF,
        training_sample: Sample,
        oob_split: np.ndarray,
        features_out: pd.Index,
        explainer_factory_fn: ExplainerFactory,
        interaction_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
    ) -> pd.DataFrame:
        x_oob = BaseShapCalculator._x_oob(model, training_sample, oob_split)

        # calculate the im values (returned as an array)
        explainer = explainer_factory_fn(model.final_estimator.root_estimator, x_oob)

        try:
            # noinspection PyUnresolvedReferences
            shap_interaction_values_fn = explainer.shap_interaction_values
        except AttributeError:
            raise RuntimeError(
                "Explainer does not implement method shap_interaction_values"
            )

        shap_interaction_tensors: Union[
            np.ndarray, List[np.ndarray]
        ] = shap_interaction_values_fn(x_oob)

        if isinstance(shap_interaction_tensors, np.ndarray):
            # if we have a single target *and* no classification, the explainer will
            # have returned a single tensor as an array, so we wrap it in a list
            shap_interaction_tensors: List[np.ndarray] = [shap_interaction_tensors]

        interaction_matrix_per_target: List[pd.DataFrame] = [
            im.reindex(
                index=pd.MultiIndex.from_product((oob_split, features_out)),
                columns=features_out,
                copy=False,
                fill_value=0.0,
            )
            for im in interaction_matrix_for_split_to_df_fn(
                shap_interaction_tensors, oob_split, x_oob.columns
            )
        ]

        # if we have a single target, use the data frame for that target;
        # else, concatenate the values data frame for all targets horizontally
        # and add a top level to the column index indicating each target
        if len(interaction_matrix_per_target) == 1:
            assert training_sample.n_targets == 1
            return interaction_matrix_per_target[0]
        else:
            assert training_sample.n_targets > 1
            return pd.concat(
                interaction_matrix_per_target,
                axis=1,
                keys=training_sample.target_columns,
                names=[Sample.COL_TARGET],
            )


class RegressorShapValuesCalculator(ShapValuesCalculator):
    """
    Calculates SHAP matrices for regression models.
    """

    @staticmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
            for raw_shap_matrix in raw_shap_tensors
        ]


class RegressorShapInteractionValuesCalculator(ShapInteractionValuesCalculator):
    """
    Calculates SHAP interaction matrices for regression models.
    """

    @staticmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        row_index = pd.MultiIndex.from_product((observations, features_in_split))

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


class ClassifierShapValuesCalculator(ShapValuesCalculator):
    """
    Calculates SHAP matrices for classification models.
    """

    @staticmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        # todo: adapt this function (and override others) to support non-binary
        #   classification

        # the shap explainer returned an array [obs x features] for each of the
        # target-classes

        n_arrays = len(raw_shap_tensors)

        # we decided to support only binary classification == 2 classes:
        assert n_arrays == 2, (
            "classification pipeline inspection only supports binary classifiers, "
            f"but SHAP analysis returned values for {n_arrays} classes"
        )

        # in the binary classification case, we will proceed with SHAP values
        # for class 0, since values for class 1 will just be the same
        # values times (*-1)  (the opposite probability)

        # to ensure the values are returned as expected above,
        # and no information of class 1 is discarded, assert the
        # following:
        assert np.allclose(
            raw_shap_tensors[0], -raw_shap_tensors[1]
        ), "shap_values(class 0) == -shap_values(class 1)"

        # all good: proceed with SHAP values for class 0:
        raw_shap_matrix = raw_shap_tensors[0]

        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
        ]


class ClassifierShapInteractionValuesCalculator(ShapInteractionValuesCalculator):
    """
    Calculates SHAP interaction matrices for classification models.
    """

    @staticmethod
    def _raw_shap_to_df(
        raw_shap_tensors: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        raise NotImplementedError(
            "interaction matrices for classifiers are not yet implemented"
        )
