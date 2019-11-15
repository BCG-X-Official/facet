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

from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import ClassifierCrossfit, LearnerCrossfit, RegressorCrossfit
from gamma.ml.inspection._shap import (
    ClassifierInteractionMatrixCalculator,
    ClassifierShapMatrixCalculator,
    ExplainerFactory,
    InteractionMatrixCalculator,
    RegressorInteractionMatrixCalculator,
    RegressorShapMatrixCalculator,
    ShapMatrixCalculator,
)
from gamma.sklearndf import BaseLearnerDF
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)
from gamma.viz.dendrogram import LinkageTree

log = logging.getLogger(__name__)

__all__ = [
    "kernel_explainer_factory",
    "tree_explainer_factory",
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

    COL_IMPORTANCE = "importance"
    COL_IMPORTANCE_MARGINAL = "marginal importance"

    def __init__(
        self,
        crossfit: LearnerCrossfit[T_LearnerPipelineDF],
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param crossfit: predictor containing the information about the
          pipeline, the data (a Sample object), the cross-validation and crossfit.
        :param explainer_factory: calibration that returns a shap Explainer
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

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

        self._shap_matrix_calculator = self._shap_matrix_calculator_cls()(
            explainer_factory=self._explainer_factory,
            n_jobs=self.n_jobs,
            shared_memory=self.shared_memory,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
        )

        self._interaction_matrix_calculator = self._interaction_matrix_calculator_cls()(
            explainer_factory=self._explainer_factory,
            n_jobs=self.n_jobs,
            shared_memory=self.shared_memory,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
        )

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
        return self._fitted_shap_matrix_calculator().matrix

    def interaction_matrix(self) -> pd.DataFrame:
        """
        Calculate the SHAP interaction matrix for all splits.

        Each row is an observation in a specific test split, and each column is a
        combination of two features. Values are the SHAP interaction values per
        observation, calculated as the mean SHAP interaction value across all splits
        that contain the observation.

        :return: SHAP interaction matrix as a data frame
        """
        return self._fitted_interaction_matrix_calculator().matrix

    def feature_importance(
        self, *, marginal: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Feature importance computed using absolute value of shap values.

        :param marginal: if `True` calculate marginal feature importance, i.e., \
            the relative loss in SHAP contribution when the feature is removed \
            and all other features remain in place (default: `False`)

        :return: feature importances as their mean absolute SHAP contributions, \
          normalised to a total 100%. Returned as a series of length n_features for \
          single-target models, and as a data frame of shape (n_features, n_targets) \
          for multi-target models
        """
        shap_matrix = self.shap_matrix()
        mean_abs_importance: pd.Series = shap_matrix.abs().mean()

        total_importance: float = mean_abs_importance.sum()

        if marginal:

            diagonals = self._fitted_interaction_matrix_calculator().diagonals()

            # noinspection PyTypeChecker
            mean_abs_importance_marginal: pd.Series = (
                cast(pd.DataFrame, shap_matrix * 2) - diagonals
            ).abs().mean()

            # noinspection PyTypeChecker
            feature_importances = cast(
                pd.Series, mean_abs_importance_marginal / total_importance
            ).rename(BaseLearnerInspector.COL_IMPORTANCE_MARGINAL)

        else:
            # noinspection PyTypeChecker
            feature_importances = cast(
                pd.Series, mean_abs_importance / total_importance
            ).rename(BaseLearnerInspector.COL_IMPORTANCE)

        if self._n_targets > 1:
            assert (
                mean_abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            feature_importances: pd.DataFrame = mean_abs_importance.unstack(level=0)

        return feature_importances

    def feature_association_matrix(self) -> pd.DataFrame:
        """
        Return the Pearson correlation matrix of the shap matrix.

        :return: data frame with column and index given by the feature names,
          and values are the Pearson correlations of the shap values of features
        """
        if self._feature_dependency_matrix is None:
            self._feature_dependency_matrix = self._feature_matrix_to_df(
                self._association_matrix()
            )

        return self._feature_dependency_matrix

    def feature_association_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Return the :class:`.LinkageTree` based on the
        :meth:`.feature_dependency_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        return self._linkage_from_affinity_matrix(
            feature_affinity_matrix=self._association_matrix()
        )

    @property
    def _n_targets(self) -> int:
        return self.crossfit.training_sample.n_targets

    @property
    def _features(self) -> pd.Index:
        return self.crossfit.base_estimator.features_out.rename(Sample.COL_FEATURE)

    @property
    def _n_features(self) -> int:
        return len(self._features)

    def _feature_matrix_to_df(self, matrix: np.ndarray) -> pd.DataFrame:
        n_features = self._n_features
        n_targets = self._n_targets

        # transform to 2D shape (n_features, n_targets * n_features)
        matrix_2d = matrix.swapaxes(0, 1).reshape((n_features, n_targets * n_features))

        # convert ndarray to data frame with appropriate indices
        matrix_df = pd.DataFrame(
            data=matrix_2d, columns=self.shap_matrix().columns, index=self._features
        )

        assert matrix_df.shape == (n_features, n_targets * n_features)

        return matrix_df

    def _linkage_from_affinity_matrix(
        self, feature_affinity_matrix: np.ndarray
    ) -> Union[LinkageTree, List[LinkageTree]]:
        # calculate the linkage trees for all targets in a feature distance matrix;
        # matrix has shape (n_targets, n_features, n_features) with values ranging from
        # (1 = closest, 0 = most distant)
        # return a linkage tree if there is only one target, else return a list of
        # linkage trees
        n_targets = feature_affinity_matrix.shape[0]
        if n_targets == 1:
            return self._linkage_from_affinity_matrix_for_target(
                feature_affinity_matrix, target=0
            )
        else:
            return [
                self._linkage_from_affinity_matrix_for_target(
                    feature_affinity_matrix, target=i
                )
                for i in range(n_targets)
            ]

    def _linkage_from_affinity_matrix_for_target(
        self, feature_affinity_matrix: np.ndarray, target: int
    ) -> LinkageTree:
        # calculate the linkage tree from the a given target in a feature distance
        # matrix;
        # matrix has shape (n_targets, n_features, n_features) with values ranging from
        # (1 = closest, 0 = most distant)
        # arg target is an integer index

        # compress the distance matrix (required by SciPy)
        compressed_distance_vector = squareform(
            1 - abs(feature_affinity_matrix[target])
        )

        # calculate the linkage matrix
        linkage_matrix = linkage(y=compressed_distance_vector, method="single")

        # Feature labels and weights will be used as the leaves of the linkage tree.
        # Select only the features that appear in the distance matrix, and in the
        # correct order

        feature_importance = self.feature_importance()
        n_targets = self._n_targets

        # build and return the linkage tree
        return LinkageTree(
            scipy_linkage_matrix=linkage_matrix,
            leaf_labels=feature_importance.index,
            leaf_weights=feature_importance.values[n_targets]
            if n_targets > 1
            else feature_importance.values,
            max_distance=1.0,
        )

    def _fitted_shap_matrix_calculator(self) -> ShapMatrixCalculator:
        if not self._shap_matrix_calculator.is_fitted:
            self._shap_matrix_calculator.fit(crossfit=self.crossfit)
        return self._shap_matrix_calculator

    def _fitted_interaction_matrix_calculator(self) -> InteractionMatrixCalculator:
        if not self._interaction_matrix_calculator.is_fitted:
            self._interaction_matrix_calculator.fit(crossfit=self.crossfit)
        return self._interaction_matrix_calculator

    def _pearson_quotients(self) -> np.ndarray:
        # return an ndarray with pearson quotients for all feature pairs
        # with shape (n_targets, n_features, n_features)

        n_targets = self._n_targets
        n_features = self._n_features

        # calculate the shap variance for each target/feature across observations
        shap_var_per_target = (
            self.shap_matrix().values.var(axis=0).reshape((n_targets, n_features, 1))
        )

        # calculate the pearson quotients for each feature pair as the square root of
        # the product of the two features' variances
        pearson_quotients = np.sqrt(
            shap_var_per_target * np.transpose(shap_var_per_target, axes=(0, 2, 1))
        )

        # this should give us one quotients matrix per target
        assert pearson_quotients.shape == (
            n_targets,
            n_features,
            n_features,
        ), "shape of pearson quotients is (n_targets, n_features, n_features)"

        # fill the diagonals for all targets with nan

        # create a tall array so we can handle diagonals separately for each target
        pearson_quotients = pearson_quotients.reshape(
            (n_targets * n_features, n_features)
        )
        # fill the diagonals in the tall matrix, with wrap-around
        np.fill_diagonal(pearson_quotients, val=np.nan, wrap=True)
        # restore the original shape
        pearson_quotients = pearson_quotients.reshape(
            (n_targets, n_features, n_features)
        )

        return pearson_quotients

    def _association_matrix(self) -> np.ndarray:
        # return an ndarray with a pearson correlation matrix of the shap matrix
        # for each target, with shape (n_targets, n_features, n_features)

        n_targets: int = self._n_targets
        n_features: int = self._n_features

        # get the shap matrix as an ndarray of shape
        # (n_targets, n_observations, n_features);
        # this is achieved by re-shaping the shap matrix to get the additional "target"
        # dimension, then swapping the target and observation dimensions
        shap_matrix_per_target = (
            self.shap_matrix()
            .values.reshape((-1, n_targets, n_features))
            .swapaxes(0, 1)
        )

        def _ensure_diagonality(matrix: np.ndarray) -> np.ndarray:
            # remove potential floating point errors
            matrix = (matrix + matrix.T) / 2

            # replace nan values with 0.0 = no association when correlation is undefined
            np.nan_to_num(matrix, copy=False)

            # set the matrix diagonals to 1.0 = full association of each feature with
            # itself
            np.fill_diagonal(matrix, 1.0)

            return matrix

        # calculate the shap correlation matrix for each target, and stack matrices
        # horizontally
        return np.array(
            [
                _ensure_diagonality(np.corrcoef(shap_for_target, rowvar=False))
                for shap_for_target in shap_matrix_per_target
            ]
        )

    @staticmethod
    @abstractmethod
    def _shap_matrix_calculator_cls() -> Type[ShapMatrixCalculator]:
        pass

    @staticmethod
    @abstractmethod
    def _interaction_matrix_calculator_cls() -> Type[InteractionMatrixCalculator]:
        pass


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
    def _shap_matrix_calculator_cls() -> Type[ShapMatrixCalculator]:
        return RegressorShapMatrixCalculator

    @staticmethod
    def _interaction_matrix_calculator_cls() -> Type[InteractionMatrixCalculator]:
        return RegressorInteractionMatrixCalculator


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
    def _shap_matrix_calculator_cls() -> Type[ShapMatrixCalculator]:
        return ClassifierShapMatrixCalculator

    @staticmethod
    def _interaction_matrix_calculator_cls() -> Type[InteractionMatrixCalculator]:
        return ClassifierInteractionMatrixCalculator


# noinspection PyUnusedLocal
def tree_explainer_factory(model: BaseLearnerDF, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.

    :param model: estimator from which we want to compute shap values
    :param data: (ignored)
    :return: :class:`shap.TreeExplainer` if the estimator is compatible
    """
    return TreeExplainer(model=model)


def kernel_explainer_factory(model: BaseLearnerDF, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.

    :param model: estimator from which we want to compute shap values
    :param data: data used to compute the shap values
    :return: :class:`shap.TreeExplainer` if the estimator is compatible
    """
    return KernelExplainer(model=model.predict, data=data)


def _covariance_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # calculate the pair-wise variance of 2D matrices a and b
    # a and b must have the same shape (n_rows, n_columns)
    # returns a vector of pairwise covariances of shape (n_rows)

    assert a.ndim == b.ndim == 2, "args a and b are 2D matrices"
    assert a.shape == b.shape, "args a and b have the same shape"

    # row-wise mean of input arrays, and subtract from input arrays themselves
    a_ma = a - a.mean(-1, keepdims=True)
    b_mb = b - b.mean(-1, keepdims=True)

    # calculate pair-wise covariance for each row of a and b
    return np.einsum("ij,ij->i", a_ma, b_mb) / a.shape[1]
