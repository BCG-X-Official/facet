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

from gamma.common import deprecated
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

    @deprecated(
        message="Use method feature_importance instead. "
        "This method will be removed in a future release."
    )
    def feature_importances(
        self, *, marginal: bool = False
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Deprecated. Use :meth:`.feature_importance` instead.
        """
        return self.feature_importance(marginal=marginal)

    def feature_importance(
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

        if self._n_targets > 1:
            assert (
                mean_abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            feature_importances: pd.DataFrame = mean_abs_importance.unstack(level=0)

        return feature_importances

    def feature_equivalence_matrix(self) -> pd.DataFrame:
        """
        Return the matrix indicating pair-wise feature equivalence.

        Feature equivalence ranges from 0.0 (0% = features influence the target
        fully independently of each other) to 1.0 (100% = features are fully equivalent)

        :return: feature equivalence matrix of shape \
            (n_features, n_targets * n_features)
        """
        return self._affinity_matrix_to_df(matrix=self._equivalence())

    def feature_synergy_matrix(self) -> pd.DataFrame:
        """
        Return the matrix indicating pair-wise feature synergies.

        Feature synergies range from 0.0 (0% = no synergy among two features)
        to 1.0 (100% = full synergy among two features)

        :return: feature synergy matrix of shape \
            (n_features, n_targets * n_features)
        """
        return self._affinity_matrix_to_df(matrix=self._synergies())

    def feature_dependency_matrix(self) -> pd.DataFrame:
        """
        Return the Pearson correlation matrix of the shap matrix.

        :return: data frame with column and index given by the feature names,
          and values are the Pearson correlations of the shap values of features
        """
        if self._feature_dependency_matrix is None:
            self._feature_dependency_matrix = self._affinity_matrix_to_df(
                self._dependencies()
            )

        return self._feature_dependency_matrix

    @deprecated(
        message="Replaced by method feature_dependency_matrix. "
        "Method cluster_dependent_features will be removed in the next release."
    )
    def cluster_dependent_features(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Deprecated. Use :meth:`~.feature_dependency_linkage` instead.
        """
        return self.feature_dependency_linkage()

    def feature_dependency_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Return the :class:`.LinkageTree` based on the
        :meth:`.feature_dependency_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        return self._linkage_from_affinity_matrix(
            feature_affinity_matrix=self._dependencies(), target=0
        )

    def feature_equivalence_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Return the :class:`.LinkageTree` based on the
        :meth:`feature_equivalence_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        return self._linkage_from_affinity_matrix(
            feature_affinity_matrix=self._equivalence(), target=0
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

    def _tidy_up_affinity_matrix(self, matrix: np.ndarray) -> np.ndarray:
        # ensure exact diagonality of a calculated affinity matrix,
        # ensuring that the resulting sub-matrices (axes 1 and 2) are symmetrical
        # and setting the diagonal to 1.0 (maximum affinity);
        # matrix has shape (n_targets, n_features, n_features)

        n_features = self._n_features
        n_targets = self._n_targets

        assert matrix.shape == (n_targets, n_features, n_features)

        # ensure matrix is symmetric by mirroring upper triangle to lower triangle
        # for each target
        matrix_tidy = (matrix + np.transpose(matrix, axes=(0, 2, 1))) / 2.0

        # make sure the tidy matrix is the same as the original one
        # except for very small rounding errors
        assert np.allclose(
            matrix, matrix_tidy, atol=1e-8, rtol=1e-8, equal_nan=True
        ), f"arg matrix is diagonal:\n{matrix}\n{matrix_tidy}\n{matrix - matrix_tidy}"

        # set the matrix diagonals to 1.0 = full affinity of each feature with itself
        for matrix_for_target in matrix_tidy:
            np.fill_diagonal(matrix_for_target, 1.0)

        return matrix_tidy

    def _affinity_matrix_to_df(self, matrix: np.ndarray) -> pd.DataFrame:
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
        self, feature_affinity_matrix: np.ndarray, target: int
    ):
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

    def _dependencies(self) -> np.ndarray:
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

        # calculate the shap correlation matrix for each target, and stack matrices
        # horizontally
        dependencies_per_target = np.array(
            [
                np.corrcoef(shap_for_target, rowvar=False)
                for shap_for_target in shap_matrix_per_target
            ]
        )

        return self._tidy_up_affinity_matrix(dependencies_per_target)

    def _synergies(self) -> np.ndarray:
        # return an ndarray with feature/feature synergies for all feature pairs
        # with shape (n_targets, n_features, n_features)

        n_targets = self._n_targets
        n_features: int = self._n_features

        # calculate the variance for each target and feature/feature interaction,
        # across observations
        interaction_variance_matrix_per_target = (
            self.interaction_matrix()
            .values.reshape((-1, n_targets, n_features, n_features))
            .var(axis=0)
        )

        # get the pearson quotients
        pearson_quotients = self._pearson_quotients()

        # interaction variances and pearson quotients should both have the same shape
        assert interaction_variance_matrix_per_target.shape == (
            n_targets,
            n_features,
            n_features,
        ), (
            "shape of interaction variance matrix is "
            "(n_targets, n_features, n_features)"
        )
        assert (
            interaction_variance_matrix_per_target.shape == pearson_quotients.shape
        ), (
            "shape of interaction variance matrix and pearson quotient matrix "
            "is the same"
        )

        # the synergies are the interaction variances normalised with the pearson
        # quotients
        return self._tidy_up_affinity_matrix(
            interaction_variance_matrix_per_target / pearson_quotients
        )

    def _equivalence(self) -> np.ndarray:
        # return an ndarray with feature/feature equivalence for all feature pairs
        # with shape (n_targets, n_features, n_features)

        # equivalence is the residual dependency after subtracting synergies
        equivalence_matrix = self._dependencies() - self._synergies()

        # reset the diagonal to 1: every feature is fully equivalent with itself
        for m in equivalence_matrix:
            np.fill_diagonal(m, 1.0)

        return equivalence_matrix

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
