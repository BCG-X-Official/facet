"""
Core implementation of :mod:`gamma.ml.inspection`
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer

from gamma.common import deprecated
from gamma.common.parallelization import ParallelizableMixin
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
        self._fitted_interaction_matrix_calculator()

        return self._interaction_matrix_calculator.matrix

    def feature_importances(
        self, *, marginal: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Feature importance computed using absolute value of shap values.

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

        if self.crossfit.training_sample.n_targets > 1:
            assert (
                mean_abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            feature_importances: pd.DataFrame = mean_abs_importance.unstack(level=0)

        return feature_importances

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

    def _fitted_shap_matrix_calculator(self) -> ShapMatrixCalculator:
        if not self._shap_matrix_calculator.is_fitted:
            self._shap_matrix_calculator.fit(crossfit=self.crossfit)
        return self._shap_matrix_calculator

    def _fitted_interaction_matrix_calculator(self) -> InteractionMatrixCalculator:
        if not self._interaction_matrix_calculator.is_fitted:
            self._interaction_matrix_calculator.fit(crossfit=self.crossfit)
        return self._interaction_matrix_calculator

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


def tree_explainer_factory(model: BaseLearnerDF, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.

    :param model: estimator from which we want to compute shap values
    :param data: data used to compute the shap values
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
