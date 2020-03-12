"""
Core implementation of :mod:`gamma.ml.inspection`
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer

from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection._shap import (
    ClassifierShapInteractionValuesCalculator,
    ClassifierShapValuesCalculator,
    ExplainerFactory,
    RegressorShapInteractionValuesCalculator,
    RegressorShapValuesCalculator,
    ShapCalculator,
    ShapInteractionValuesCalculator,
    ShapValuesCalculator,
)
from gamma.ml.inspection._shap_decomposition import (
    ShapInteractionValueDecomposer,
    ShapValueDecomposer,
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


class BaseLearnerInspector(
    FittableMixin[Sample], ParallelizableMixin, ABC, Generic[T_LearnerPipelineDF]
):
    """
    Inspect features interactions in a learner pipeline through SHAP values.
    """

    COL_IMPORTANCE = "importance"
    COL_IMPORTANCE_MARGINAL = "marginal importance"

    def __init__(
        self,
        *,
        explainer_factory: Optional[ExplainerFactory] = None,
        shap_interaction: bool = True,
        min_direct_synergy: Optional[float] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param explainer_factory: optional function that creates a shap Explainer \
            (default: :func:``.tree_explainer_factory``)
        :param shap_interaction: if `True`, calculate SHAP interaction values, else \
            only calculate SHAP contribution values.\
            SHAP interaction values are needed to determine feature synergy and \
            redundancy; otherwise only SHAP association can be calculated.\
            (default: `True`)
        :param min_direct_synergy: minimum direct synergy to consider a feature pair \
            for calculation of indirect synergy. \
            Only relevant if parameter `shap_interaction` is `True`. \
            (default: <DEFAULT_MIN_DIRECT_SYNERGY>)
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self._explainer_factory = (
            explainer_factory
            if explainer_factory is not None
            else tree_explainer_factory
        )
        self._shap_interaction = shap_interaction
        self._min_direct_synergy = min_direct_synergy
        self._random_state = random_state

        self._crossfit: Optional[LearnerCrossfit[T_LearnerPipelineDF]] = None
        self._shap_calculator: Optional[ShapCalculator] = None
        self._shap_decomposer: Optional[ShapValueDecomposer] = None

    # dynamically complete the __init__ docstring
    # noinspection PyTypeChecker
    __init__.__doc__ = (
        __init__.__doc__.replace(
            "<DEFAULT_MIN_DIRECT_SYNERGY>",
            str(ShapInteractionValueDecomposer.DEFAULT_MIN_DIRECT_SYNERGY),
        )
        + ParallelizableMixin.__init__.__doc__
    )

    def fit(self: T_Self, crossfit: LearnerCrossfit, **fit_params) -> T_Self:
        """
        Fit the inspector with the given sample.

        :param crossfit: the model crossfit to be explained during model inspection
        :param fit_params: additional keyword arguments (ignored)
        :return: `self`
        """

        self: BaseLearnerInspector  # support type hinting in PyCharm

        if not crossfit.is_fitted:
            raise ValueError("crossfit in arg pipeline is not fitted")

        if self._shap_interaction:
            shap_calculator = self._shap_interaction_values_calculator_cls()(
                explainer_factory=self._explainer_factory,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )
            shap_decomposer = ShapInteractionValueDecomposer(
                min_direct_synergy=self._min_direct_synergy
            )

        else:
            shap_calculator = self._shap_values_calculator_cls()(
                explainer_factory=self._explainer_factory,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )
            shap_decomposer = ShapValueDecomposer()

        shap_calculator.fit(crossfit=crossfit)
        shap_decomposer.fit(shap_calculator=shap_calculator)

        self._shap_calculator = shap_calculator
        self._shap_decomposer = shap_decomposer
        self._crossfit = crossfit

        return self

    @property
    def is_fitted(self) -> bool:
        """`True` if this inspector is fitted, else `False`"""
        return self._crossfit is not None

    @property
    def crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        CV fit of the pipeline being examined by this inspector.
        """
        self._ensure_fitted()
        return self._crossfit

    @property
    def training_sample(self) -> Sample:
        """
        The training sample used for model inspection.
        """
        self._ensure_fitted()
        return self.crossfit.training_sample

    def shap_values(self) -> pd.DataFrame:
        """
        Calculate the SHAP values for all splits.

        Each row is an observation in a specific test split, and each column is a
        feature. Values are the SHAP values per observation, calculated as the mean
        SHAP value across all splits that contain the observation.

        :return: shap values as a data frame
        """
        self._ensure_fitted()
        return self._shap_calculator.shap_values

    def shap_interaction_values(self) -> pd.DataFrame:
        """
        Calculate the SHAP interaction values for all splits.

        Each row is an observation in a specific test split, and each column is a
        combination of two features. Values are the SHAP interaction values per
        observation, calculated as the mean SHAP interaction value across all splits
        that contain the observation.

        :return: SHAP interaction values as a data frame
        """
        self._ensure_fitted()
        return self._shap_interaction_values_calculator.shap_interaction_values

    def feature_importance(
        self, *, method: str = "rms"
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Feature importance computed using relative absolute shap contributions across
        all observations.

        :param method: method for calculating feature importance. Supported methods \
            are `rms` (root of mean squares, default) and `mav` (mean absolute values)
        :return: importance of each feature as its mean absolute SHAP contribution, \
          normalised to a total 100%. Returned as a series of length n_features for \
          single-target models, and as a data frame of shape (n_features, n_targets) \
          for multi-target models
        """

        methods = ["rms", "mav"]
        if method not in methods:
            raise ValueError(
                f'arg method="{method}" must be one of {{{", ".join(methods)}}}'
            )

        shap_matrix = self.shap_values()
        abs_importance: pd.Series
        if method == "rms":
            abs_importance = shap_matrix.pow(2).mean().pow(0.5)
        elif method == "mav":
            abs_importance = shap_matrix.abs().mean()
        else:
            raise ValueError(f"unknown method: {method}")

        total_importance: float = abs_importance.sum()

        feature_importance_sr: pd.Series = abs_importance.divide(
            total_importance
        ).rename(BaseLearnerInspector.COL_IMPORTANCE)

        if self._n_targets > 1:
            assert (
                abs_importance.index.nlevels == 2
            ), "2 index levels in place for multi-output models"

            feature_importance_sr: pd.DataFrame = abs_importance.unstack(level=0)

        return feature_importance_sr

    def feature_association_matrix(self) -> pd.DataFrame:
        """
        Calculate the Pearson correlation matrix of the shap values.

        :return: data frame with column and index given by the feature names,
          and values as the Pearson correlations of the shap values of features
        """
        self._ensure_fitted()

        return self._shap_decomposer.association

    def feature_association_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate the :class:`.LinkageTree` based on the
        :meth:`.feature_association_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        self._ensure_fitted()
        return self._linkage_from_affinity_matrix(
            feature_affinity_matrix=self._shap_decomposer.association_rel_
        )

    def feature_synergy_matrix(self) -> pd.DataFrame:
        """
        For each pairing of features, calculate the relative share of their synergistic
        contribution to the model prediction.

        The synergistic contribution of a pair of features ranges between 0.0
        (no synergy - both features contribute fully autonomously) and 1.0
        (full synergy - both features combine all of their information into a joint
        contribution).

        :return: feature synergy matrix as a data frame of shape \
            (n_features, n_targets * n_features)
        """
        self._ensure_fitted()
        return self._shap_interaction_decomposer.synergy

    def feature_redundancy_matrix(self) -> pd.DataFrame:
        """
        For each pairing of features, calculate the relative share of their redundant
        contribution to the model prediction.

        The redundant contribution of a pair of features ranges between 0.0
        (no redundancy - both features contribute fully independently) and 1.0
        (full redundancy - the information used by either feature is fully redundant).

        :return: feature redundancy matrix as a data frame of shape \
            (n_features, n_targets * n_features)
        """
        self._ensure_fitted()
        return self._shap_interaction_decomposer.redundancy

    def feature_redundancy_linkage(self) -> Union[LinkageTree, List[LinkageTree]]:
        """
        Calculate the :class:`.LinkageTree` based on the
        :meth:`.feature_redundancy_matrix`.

        :return: linkage tree for the shap clustering dendrogram; \
            list of linkage trees if the base estimator is a multi-output model
        """
        self._ensure_fitted()
        return self._linkage_from_affinity_matrix(
            feature_affinity_matrix=self._shap_interaction_decomposer.redundancy_rel_
        )

    def feature_interaction_matrix(self) -> pd.DataFrame:
        """
        Calculate average shap interaction values for all feature pairings.

        Shap interactions quantify direct interactions between pairs of features.
        For a quantification of overall interaction (including indirect interactions
        among more than two features), see :meth:`.feature_synergy_matrix`.

        The average values are normalised to add up to 1.0, and each value ranges
        between 0.0 and 1.0.

        For features :math:`f_i` and :math:`f_j`, the average shap interaction is
        calculated as

        .. math::
            \\mathrm{si}_{ij} = \\frac
                {\\sigma(\\vec{\\phi}_{ij})}
                {\\sum_{a=1}^n \\sum_{b=1}^n \\sigma(\\vec{\\phi}_{ab})}

        where :math:`\\sigma(\\vec v)` is the standard deviation of all elements of
        vector :math:`\\vec v`.

        The total average interaction of features
        :math:`f_i` and :math:`f_j` is
        :math:`\\mathrm{si}_{ij} \
            + \\mathrm{si}_{ji} \
            = 2\\mathrm{si}_{ij}`.

        :math:`\\mathrm{si}_{ii}` is the residual, non-synergistic contribution
        of feature :math:`f_i`

        The matrix returned by this method is a diagonal matrix

        .. math::

            \\newcommand\\si[1]{\\mathrm{si}_{#1}}
            \\newcommand\\nan{\\mathit{nan}}
            \\si{} = \\begin{pmatrix}
                \\si{11} & \\nan & \\nan & \\dots & \\nan \\\\
                2\\si{21} & \\si{22} & \\nan & \\dots & \\nan \\\\
                2\\si{31} & 2\\si{32} & \\si{33} & \\dots & \\nan \\\\
                \\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
                2\\si{n1} & 2\\si{n2} & 2\\si{n3} & \\dots & \\si{nn} \\\\
            \\end{pmatrix}

        with :math:`\\sum_{a=1}^n \\sum_{b=a}^n \\mathrm{si}_{ab} = 1`

        :return: average shap interaction values as a data frame of shape \
            (n_features, n_targets * n_features)
        """

        n_features = self._n_features
        n_targets = self._n_targets

        # get a feature interaction array with shape
        # (n_observations, n_targets, n_features, n_features)
        # where the innermost feature x feature arrays are symmetrical
        im_matrix_per_observation_and_target = (
            self.shap_interaction_values()
            .values.reshape((-1, n_features, n_targets, n_features))
            .swapaxes(1, 2)
        )

        # calculate the average interactions for each target and feature/feature
        # interaction, based on the standard deviation assuming a mean of 0.0.
        # The resulting matrix has shape (n_targets, n_features, n_features)
        interaction_matrix = np.sqrt(
            (
                im_matrix_per_observation_and_target
                * im_matrix_per_observation_and_target
            ).mean(axis=0)
        )
        assert interaction_matrix.shape == (n_targets, n_features, n_features)

        # we normalise the synergy matrix for each target to a total of 1.0
        interaction_matrix /= interaction_matrix.sum()

        # the total interaction effect for features i and j is the total of matrix
        # cells (i,j) and (j,i); theoretically both should be the same but to minimize
        # numerical errors we total both in the lower matrix triangle (but excluding the
        # matrix diagonal, hence k=1)
        interaction_matrix += np.triu(interaction_matrix, k=1).swapaxes(1, 2)

        # discard the upper matrix triangle by setting it to nan
        interaction_matrix += np.triu(
            np.full(shape=(n_features, n_features), fill_value=np.nan), k=1
        )[np.newaxis, :, :]

        # create a data frame from the feature matrix
        return self._feature_matrix_to_df(interaction_matrix)

    @property
    def _n_targets(self) -> int:
        return self.training_sample.n_targets

    @property
    def _features(self) -> pd.Index:
        return self.crossfit.pipeline.features_out.rename(Sample.COL_FEATURE)

    @property
    def _n_features(self) -> int:
        return len(self._features)

    def _feature_matrix_to_df(self, matrix: np.ndarray) -> pd.DataFrame:
        # transform a matrix of shape (n_targets, n_features, n_features)
        # to a data frame

        n_features = self._n_features
        n_targets = self._n_targets

        assert matrix.shape == (n_targets, n_features, n_features)

        # transform to 2D shape (n_features, n_targets * n_features)
        matrix_2d = matrix.swapaxes(0, 1).reshape((n_features, n_targets * n_features))

        # convert array to data frame with appropriate indices
        matrix_df = pd.DataFrame(
            data=matrix_2d, columns=self.shap_values().columns, index=self._features
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

    def _ensure_shap_interaction(self) -> None:
        if not self._shap_interaction:
            raise RuntimeError(
                "SHAP interaction values have not been calculated. "
                "Create an inspector with parameter 'shap_interaction=True' to "
                "enable calculations involving SHAP interaction values."
            )

    @property
    def _shap_interaction_values_calculator(self) -> ShapInteractionValuesCalculator:
        self._ensure_shap_interaction()
        return cast(ShapInteractionValuesCalculator, self._shap_calculator)

    @property
    def _shap_interaction_decomposer(self) -> ShapInteractionValueDecomposer:
        self._ensure_shap_interaction()
        return cast(ShapInteractionValueDecomposer, self._shap_decomposer)

    @staticmethod
    @abstractmethod
    def _shap_values_calculator_cls() -> Type[ShapValuesCalculator]:
        pass

    @staticmethod
    @abstractmethod
    def _shap_interaction_values_calculator_cls() -> Type[
        ShapInteractionValuesCalculator
    ]:
        pass


class RegressorInspector(
    BaseLearnerInspector[T_RegressorPipelineDF], Generic[T_RegressorPipelineDF]
):
    """
    Inspect a regression pipeline through its SHAP values.
    """

    @staticmethod
    def _shap_values_calculator_cls() -> Type[ShapValuesCalculator]:
        return RegressorShapValuesCalculator

    @staticmethod
    def _shap_interaction_values_calculator_cls() -> Type[
        ShapInteractionValuesCalculator
    ]:
        return RegressorShapInteractionValuesCalculator


class ClassifierInspector(
    BaseLearnerInspector[T_ClassifierPipelineDF], Generic[T_ClassifierPipelineDF]
):
    """
    Inspect a classification pipeline through its SHAP values.

    Currently only binary, single-output classifiers are supported.
    """

    @staticmethod
    def _shap_values_calculator_cls() -> Type[ShapValuesCalculator]:
        return ClassifierShapValuesCalculator

    @staticmethod
    def _shap_interaction_values_calculator_cls() -> Type[
        ShapInteractionValuesCalculator
    ]:
        return ClassifierShapInteractionValuesCalculator


# noinspection PyUnusedLocal
def tree_explainer_factory(model: BaseLearnerDF, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.

    :param model: estimator from which we want to compute shap values
    :param data: background dataset (ignored)
    :return: :class:`shap.TreeExplainer` if the estimator is compatible
    """
    return TreeExplainer(model=model)


def kernel_explainer_factory(model: BaseLearnerDF, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.

    :param model: estimator from which we want to compute shap values
    :param data: background dataset
    :return: :class:`shap.TreeExplainer` if the estimator is compatible
    """
    return KernelExplainer(model=model.predict, data=data)
