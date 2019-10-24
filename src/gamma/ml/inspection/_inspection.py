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
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)
from gamma.viz.dendrogram import LinkageTree

log = logging.getLogger(__name__)

__all__ = ["BaseLearnerInspector", "ClassifierInspector", "RegressorInspector"]

#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)


#
# Class definitions
#


class BaseLearnerInspector(ParallelizableMixin, ABC, Generic[T_LearnerPipelineDF]):
    """
    Inspect a pipeline through its SHAP values.

    :param crossfit: predictor containing the information about the
      pipeline, the data (a Sample object), the cross-validation and crossfit.
    :param explainer_factory: calibration that returns a shap Explainer
    """

    __slots__ = [
        "_cross_fit",
        "_shap_matrix",
        "_feature_dependency_matrix",
        "_explainer_factory",
    ]

    COL_FEATURE = "feature"

    def __init__(
        self,
        crossfit: LearnerCrossfit[T_LearnerPipelineDF],
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
        *,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
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
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None

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
        feature, and values are the SHAP values per observation/split and feature.

        :return: shap matrix as a data frame
        """

        if self._shap_matrix is not None:
            return self._shap_matrix

        crossfit = self.crossfit
        shap_matrix_for_split_to_df_fn = self._shap_matrix_for_split_to_df
        explainer_factory_fn = self._explainer_factory
        features_out = (
            crossfit.base_estimator.preprocessing.features_out
            if crossfit.base_estimator.preprocessing is not None
            else crossfit.base_estimator.features_in
        )

        training_sample = crossfit.training_sample

        with self._parallel() as parallel:
            shap_values_df_for_splits = parallel(
                self._delayed(BaseLearnerInspector._shap_values_for_split)(
                    model,
                    training_sample,
                    oob_split,
                    features_out,
                    explainer_factory_fn,
                    shap_matrix_for_split_to_df_fn,
                )
                for model, (_train_split, oob_split) in zip(
                    crossfit.models(), crossfit.splits()
                )
            )

        shap_values_df = pd.concat(shap_values_df_for_splits)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = shap_values_df.groupby(by=shap_values_df.index).mean()

        return self._shap_matrix

    @staticmethod
    def _shap_values_for_split(
        model: T_LearnerPipelineDF,
        training_sample: Sample,
        oob_split: np.ndarray,
        features_out: pd.Index,
        explainer_factory_fn: Callable[[BaseEstimator, pd.DataFrame], Explainer],
        shap_matrix_for_split_to_df_fn: Callable[
            [Union[np.ndarray, List[np.ndarray]], Sequence, Sequence], pd.DataFrame
        ],
    ):

        # get the features of all out-of-bag observations
        x_oob = training_sample.subsample(loc=oob_split).features

        # pre-process the features
        if model.preprocessing is not None:
            x_oob = model.preprocessing.transform(x_oob)

        # calculate the shap values (returned as an ndarray)
        shap_values_ndarray = explainer_factory_fn(
            model.final_estimator.root_estimator, x_oob
        ).shap_values(x_oob)

        # convert to a data frame (different logic depending on whether we have a
        # regressor or a classifier)
        shap_values_df = shap_matrix_for_split_to_df_fn(
            shap_values_ndarray, oob_split, x_oob.columns
        )

        # reindex to add missing columns and fill n/a values with zero shap importance
        return shap_values_df.reindex(columns=features_out).fillna(0.0)

    @staticmethod
    @abstractmethod
    def _shap_matrix_for_split_to_df(
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        index: Sequence,
        columns: Sequence,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param index: index of the transformed data the pipeline was trained on
        :param columns: columns of the transformed data the pipeline was trained on
        :return: SHAP matrix of a single split as data frame
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

            # exclude features with zero Shapley importance
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
        feature_importances = feature_importances.loc[
            feature_importances.index.intersection(feature_distance_matrix.index)
        ]

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

    :param crossfit: regressor containing the information about the pipeline, \
        the data (a Sample object), the cross-validation and crossfit.
    :param explainer_factory: calibration that returns a shap Explainer
    """

    def __init__(
        self,
        crossfit: RegressorCrossfit[T_RegressorPipelineDF],
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
        *,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            crossfit=crossfit,
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

    @staticmethod
    def _shap_matrix_for_split_to_df(
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        index: Sequence,
        columns: Sequence,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param index: index of the transformed data the pipeline was trained on
        :param columns: columns of the transformed data the pipeline was trained on
        :return: SHAP matrix of a single split as data frame
        """

        # In the regression case, only ndarray outputs are expected from SHAP:
        if not isinstance(raw_shap_values, np.ndarray):
            raise ValueError(
                "shap explainer output expected to be an ndarray but was "
                f"{type(raw_shap_values)}"
            )

        return pd.DataFrame(data=raw_shap_values, index=index, columns=columns)


class ClassifierInspector(
    BaseLearnerInspector[T_ClassifierPipelineDF], Generic[T_ClassifierPipelineDF]
):
    """
    Inspect a classification pipeline through its SHAP values.

    Currently only binary, single-output classification problems are supported.

    :param crossfit: classifier containing the information about the pipeline, \
        the data (a Sample object), the cross-validation and crossfit.
    :param explainer_factory: calibration that returns a shap Explainer
    """

    def __init__(
        self,
        crossfit: ClassifierCrossfit[T_ClassifierPipelineDF],
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
        *,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            crossfit=crossfit,
            explainer_factory=explainer_factory,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

    @staticmethod
    def _shap_matrix_for_split_to_df(
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        index: Sequence,
        columns: Sequence,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param index: index of the transformed data the pipeline was trained on
        :param columns: columns of the transformed data the pipeline was trained on
        :return: SHAP matrix of a single split as data frame
        """

        # todo: adapt this calibration (and override others) to support non-binary
        #   classification

        if isinstance(raw_shap_values, list):
            # the shap explainer returned an array [obs x features] for each of the
            # target-classes

            n_arrays = len(raw_shap_values)

            # we decided to support only binary classification == 2 classes:
            assert n_arrays == 2, (
                "classification pipeline inspection only supports binary classifiers, "
                f"but SHAP analysis returned values for {n_arrays} classes"
            )

            # in the binary classification case, we will proceed with SHAP values
            # for class 0, since values for class 1 will just be the same
            # values times (*-1)  (the opposite probability)

            # to assure the values are returned as expected above,
            # and no information of class 1 is discarded, assert the
            # following:
            assert (
                np.allclose(raw_shap_values[0], -raw_shap_values[1]),
                ("shap_values(class 0) == -shap_values(class 1)"),
            )

            # all good: proceed with SHAP values for class 0:
            raw_shap_values = raw_shap_values[0]

        # after the above transformation, `raw_shap_values` should be ndarray:
        if not isinstance(raw_shap_values, np.ndarray):
            raise ValueError(
                f"shap explainer output expected to be an ndarray but was "
                f"{type(raw_shap_values)}"
            )

        return pd.DataFrame(data=raw_shap_values, index=index, columns=columns)
