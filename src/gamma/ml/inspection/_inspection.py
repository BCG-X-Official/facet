"""
Core implementation of :mod:`gamma.ml.inspection`
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from gamma.common import deprecated
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import ClassifierCrossfit, LearnerCrossfit, RegressorCrossfit
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)
from gamma.viz.dendrogram import LinkageTree

log = logging.getLogger(__name__)

__all__ = [
    "ExplainerFactory",
    "BaseLearnerInspector",
    "ClassifierInspector",
    "RegressorInspector",
]

#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)


#
# Type definitions
#

ExplainerFactory = Callable[[BaseEstimator, pd.DataFrame], Explainer]

ShapToDataFrameFunction = Callable[
    [List[np.ndarray], np.ndarray, pd.Index], List[pd.DataFrame]
]

ShapFunction = Callable[
    [
        T_LearnerPipelineDF,  # model
        Sample,  # training_sample
        np.ndarray,  # oob_split
        pd.Index,  # features_out
        ExplainerFactory,  # explainer_factory_fn
        ShapToDataFrameFunction,  # shap_matrix_for_split_to_df_fn
    ],
    pd.DataFrame,
]


#
# Class definitions
#


class BaseLearnerInspector(ParallelizableMixin, ABC, Generic[T_LearnerPipelineDF]):
    """
    Inspect a pipeline through its SHAP values.
    """

    __slots__ = [
        "_cross_fit",
        "_shap_matrix",
        "_feature_dependency_matrix",
        "_explainer_factory",
    ]

    def __init__(
        self,
        crossfit: LearnerCrossfit[T_LearnerPipelineDF],
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param crossfit: predictor containing the information about the
          pipeline, the data (a Sample object), the cross-validation and crossfit.
        :param explainer_factory: calibration that returns a shap Explainer
        """
        super().__init__(n_jobs=n_jobs, shared_memory=shared_memory, verbose=verbose)

        if not crossfit.is_fitted:
            raise ValueError("arg crossfit expected to be fitted")

        self._cross_fit = crossfit
        self._explainer_factory = (
            explainer_factory
            if explainer_factory is not None
            else tree_explainer_factory
        )
        self.n_jobs = n_jobs
        self.shared_memory = shared_memory
        self.verbose = verbose

        self._shap_matrix: Optional[pd.DataFrame] = None
        self._interaction_matrix: Optional[pd.DataFrame] = None
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None

    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @property
    def crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        CV fit of the pipeline being examined by this inspector
        """
        return self._cross_fit

    def shap_matrix(self) -> pd.DataFrame:
        """
        Calculate the SHAP matrix for all splits.

        Each row is an observation in a specific test split, and each column is a
        feature. Values are the SHAP values per observation, calculated as the mean
        SHAP value across all splits that contain the observation.

        :return: shap matrix as a data frame
        """

        if self._shap_matrix is not None:
            return self._shap_matrix

        shap_fn = BaseLearnerInspector._shap_matrix_for_split
        raw_shap_to_df_fn = self._shap_matrix_for_split_to_df

        shap_values_df = self._calculate_shap(shap_fn, raw_shap_to_df_fn)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = shap_values_df.groupby(by=shap_values_df.index).mean()

        return self._shap_matrix

    def interaction_matrix(self) -> pd.DataFrame:
        """
        Calculate the SHAP interaction matrix for all splits.

        Each row is an observation in a specific test split, and each column is a
        combination of two features. Values are the SHAP interaction values per
        observation, calculated as the mean SHAP interaction value across all splits
        that contain the observation.

        :return: SHAP interaction matrix as a data frame
        """
        if self._interaction_matrix is not None:
            return self._interaction_matrix

        shap_fn = BaseLearnerInspector._interaction_matrix_for_split
        raw_shap_to_df_fn = self._interaction_matrix_for_split_to_df

        interaction_values_df = self._calculate_shap(shap_fn, raw_shap_to_df_fn)

        # Group SHAP matrix by observation ID and feature, and aggregate using mean()
        self._interaction_matrix = interaction_values_df.groupby(level=(0, 1)).mean()

        return self._interaction_matrix

    def _calculate_shap(
        self, shap_fn: ShapFunction, raw_shap_to_df_fn: ShapToDataFrameFunction
    ) -> pd.DataFrame:
        crossfit = self.crossfit
        explainer_factory_fn = self._explainer_factory
        features_out: pd.Index = (
            crossfit.base_estimator.preprocessing.features_out
            if crossfit.base_estimator.preprocessing is not None
            else crossfit.base_estimator.features_in
        ).rename(Sample.COL_FEATURE)

        training_sample = crossfit.training_sample

        with self._parallel() as parallel:
            shap_df_per_split = parallel(
                self._delayed(shap_fn)(
                    model,
                    training_sample,
                    oob_split,
                    features_out,
                    explainer_factory_fn,
                    raw_shap_to_df_fn,
                )
                for model, (_train_split, oob_split) in zip(
                    crossfit.models(), crossfit.splits()
                )
            )
        shap_values_df = pd.concat(shap_df_per_split)
        return shap_values_df

    @staticmethod
    def _shap_matrix_for_split(
        model: T_LearnerPipelineDF,
        training_sample: Sample,
        oob_split: np.ndarray,
        features_out: pd.Index,
        explainer_factory_fn: ExplainerFactory,
        shap_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
    ):
        # get the features of all out-of-bag observations
        x_oob = training_sample.subsample(loc=oob_split).features

        # pre-process the features
        if model.preprocessing is not None:
            x_oob = model.preprocessing.transform(x_oob)

        # calculate the shap values (returned as an ndarray)
        shap_values = explainer_factory_fn(
            model.final_estimator.root_estimator, x_oob
        ).shap_values(x_oob)

        target = training_sample.target

        if isinstance(shap_values, np.ndarray):
            # if we have a single target, the explainer will have returned a single
            # tensor as an ndarray
            shap_values: List[np.ndarray] = [shap_values]

        if isinstance(target, pd.Series):
            target_names = [target.name]
        else:
            target_names = target.columns.values

        # convert to a data frame per target (different logic depending on whether we
        # have a regressor or a classifier)
        shap_values_df: List[pd.DataFrame] = [
            shap.reindex(columns=features_out).fillna(0.0)
            for shap in shap_matrix_for_split_to_df_fn(
                shap_values, oob_split, x_oob.columns
            )
        ]

        # if we have a single target, return that target; else, add a top level to the
        # column index indicating each target
        if len(shap_values_df) == 1:
            return shap_values_df[0]
        else:
            return pd.concat(
                shap_values_df, axis=1, keys=target_names, names=[Sample.COL_TARGET]
            )

    @staticmethod
    @abstractmethod
    def _shap_matrix_for_split_to_df(
        raw_shap_matrix: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_matrix: the raw values returned by the SHAP explainer
        :param observations: the ids used for indexing the explained observations
        :param features_in_split: the features in the current split, \
            explained by the SHAP explainer
        :return: SHAP matrix of a single split as data frame
        """
        pass

    @staticmethod
    def _interaction_matrix_for_split(
        model: T_LearnerPipelineDF,
        training_sample: Sample,
        oob_split: np.ndarray,
        features_out: pd.Index,
        explainer_factory_fn: ExplainerFactory,
        interaction_matrix_for_split_to_df_fn: ShapToDataFrameFunction,
    ) -> pd.DataFrame:
        # get the features of all out-of-bag observations
        x_oob = training_sample.subsample(loc=oob_split).features

        # pre-process the features
        if model.preprocessing is not None:
            x_oob = model.preprocessing.transform(x_oob)

        # calculate the shap values (returned as an ndarray)
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

        target = training_sample.target

        if isinstance(shap_interaction_tensors, np.ndarray):
            # if we have a single target, the explainer will have returned a single
            # tensor as an ndarray
            shap_interaction_tensors: List[np.ndarray] = [shap_interaction_tensors]

        if isinstance(target, pd.Series):
            target_names = [target.name]
        else:
            target_names = target.columns.values

        # convert to a data frame per target (different logic depending on whether we
        # have a regressor or a classifier)
        # reindex the interaction matrices to ensure all features are included
        interaction_matrix_per_target: List[pd.DataFrame] = [
            interaction_matrix_df.reindex(
                index=pd.MultiIndex.from_product(
                    iterables=(interaction_matrix_df.index.levels[0], features_out),
                    names=(training_sample.index.name, Sample.COL_FEATURE),
                ),
                columns=features_out,
            ).fillna(0.0)
            for interaction_matrix_df in interaction_matrix_for_split_to_df_fn(
                shap_interaction_tensors, oob_split, x_oob.columns
            )
        ]

        # if we have a single target, return that target; else, add a top level to the
        # column index indicating each target
        if len(interaction_matrix_per_target) == 1:
            return interaction_matrix_per_target[0]
        else:
            return pd.concat(
                interaction_matrix_per_target,
                axis=1,
                keys=target_names,
                names=[Sample.COL_TARGET],
            )

    @staticmethod
    @abstractmethod
    def _interaction_matrix_for_split_to_df(
        raw_interaction_tensors: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        """
        Convert the SHAP interaction matrix for a single split to a data frame.

        :param raw_interaction_tensors: the raw values returned by the SHAP explainer
        :param observations: the ids used for indexing the explained observations
        :param features_in_split: the features in the current split, \
            explained by the SHAP explainer
        :return: SHAP interaction matrix of a single split as data frame
        """
        pass

    def feature_importances(self) -> pd.Series:
        """
        Feature importance computed using absolute value of shap values.

        :return: feature importances as their mean absolute SHAP contributions,
          normalised to a total 100%
        """
        feature_importances: pd.Series = self.shap_matrix().abs().mean()
        return (feature_importances / feature_importances.sum()).sort_values(
            ascending=False
        )

    def feature_dependency_matrix(self) -> pd.DataFrame:
        """
        Return the Pearson correlation matrix of the shap matrix.

        :return: data frame with column and index given by the feature names,
          and values are the Pearson correlations of the shap values of features
        """
        if self._feature_dependency_matrix is None:
            shap_matrix = self.shap_matrix()

            # exclude features with zero SHAP importance
            # noinspection PyUnresolvedReferences
            shap_matrix = shap_matrix.loc[:, (shap_matrix != 0.0).any()]

            self._feature_dependency_matrix = shap_matrix.corr(method="pearson")

        return self._feature_dependency_matrix

    @deprecated(
        message="method cluster_dependent_features has been replaced by method "
        "feature_dependency_linkage and will be removed in a future version."
    )
    def cluster_dependent_features(self) -> LinkageTree:
        """
        Deprecated. Use :meth:`~.feature_dependency_linkage` instead.
        """
        return self.feature_dependency_linkage()

    def feature_dependency_linkage(self) -> LinkageTree:
        """
        Return the :class:`.LinkageTree` based on the `feature_dependency_matrix`.

        :return: linkage tree for the shap clustering dendrogram
        """
        # convert shap correlations to distances (1 = most distant)
        feature_distance_matrix = 1 - self.feature_dependency_matrix().abs()

        # compress the distance matrix (required by SciPy)
        compressed_distance_vector = squareform(feature_distance_matrix)

        # calculate the linkage matrix
        linkage_matrix = linkage(y=compressed_distance_vector, method="single")

        # feature labels and weights will be used as the leaves of the linkage tree
        feature_importances = self.feature_importances()

        # select only the features that appear in the distance matrix, and in the
        # correct order
        feature_importances = feature_importances.reindex(feature_distance_matrix.index)

        # build and return the linkage tree
        return LinkageTree(
            scipy_linkage_matrix=linkage_matrix,
            leaf_labels=feature_importances.index,
            leaf_weights=feature_importances.values,
            max_distance=1.0,
        )


def tree_explainer_factory(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.
    Otherwise return :class:`shap.KernelExplainer` which is expected to be much slower.

    :param estimator: estimator from which we want to compute shap values
    :param data: data used to compute the shap values
    :return: :class:`shap.TreeExplainer` if the estimator is compatible,
        else :class:`shap.KernelExplainer`."""

    # NOTE:
    # unfortunately, there is no convenient function in shap to determine the best
    # explainer calibration. hence we use this try/except approach.
    # further there is no consistent "ModelPipelineDF type X is unsupported"
    # exception raised,
    # which is why we need to always assume the error resulted from this cause -
    # we should not attempt to filter the exception type or message given that it is
    # currently inconsistent

    try:
        return TreeExplainer(model=estimator)
    except Exception as e:
        log.debug(
            f"failed to instantiate shap.TreeExplainer:{str(e)},"
            "using shap.KernelExplainer as fallback"
        )
        # when using KernelExplainer, shap expects "pipeline" to be a callable that
        # predicts
        # noinspection PyUnresolvedReferences
        return KernelExplainer(model=estimator.predict, data=data)


class RegressorInspector(
    BaseLearnerInspector[T_RegressorPipelineDF], Generic[T_RegressorPipelineDF]
):
    """
    Inspect a regression pipeline through its SHAP values.
    """

    def __init__(
        self,
        crossfit: RegressorCrossfit[T_RegressorPipelineDF],
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param crossfit: regressor containing the information about the pipeline, \
            the data (a Sample object), the cross-validation and crossfit.
        :param explainer_factory: calibration that returns a shap Explainer
        """
        super().__init__(
            crossfit=crossfit,
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @staticmethod
    def _shap_matrix_for_split_to_df(
        raw_shap_matrices: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:

        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
            for raw_shap_matrix in raw_shap_matrices
        ]

    @staticmethod
    def _interaction_matrix_for_split_to_df(
        raw_interaction_tensors: List[np.ndarray],
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
            for raw_interaction_tensor in raw_interaction_tensors
        ]


class ClassifierInspector(
    BaseLearnerInspector[T_ClassifierPipelineDF], Generic[T_ClassifierPipelineDF]
):
    """
    Inspect a classification pipeline through its SHAP values.

    Currently only binary, single-output classification problems are supported.
    """

    def __init__(
        self,
        crossfit: ClassifierCrossfit[T_ClassifierPipelineDF],
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param crossfit: classifier containing the information about the pipeline, \
            the data (a Sample object), the cross-validation and crossfit.
        :param explainer_factory: function that returns a shap Explainer
        """
        super().__init__(
            crossfit=crossfit,
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @staticmethod
    def _shap_matrix_for_split_to_df(
        raw_shap_matrix: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:

        # todo: adapt this function (and override others) to support non-binary
        #   classification

        # the shap explainer returned an array [obs x features] for each of the
        # target-classes

        n_arrays = len(raw_shap_matrix)

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
            raw_shap_matrix[0], -raw_shap_matrix[1]
        ), "shap_values(class 0) == -shap_values(class 1)"

        # all good: proceed with SHAP values for class 0:
        raw_shap_matrix = raw_shap_matrix[0]

        return [
            pd.DataFrame(
                data=raw_shap_matrix, index=observations, columns=features_in_split
            )
        ]

    @staticmethod
    def _interaction_matrix_for_split_to_df(
        raw_interaction_tensors: List[np.ndarray],
        observations: np.ndarray,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        raise NotImplementedError(
            "interaction matrices for classifiers are not yet implemented"
        )
