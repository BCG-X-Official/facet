"""
Projection of SHAP contribution scores (i.e, SHAP importance) of all possible
pairings of features onto the SHAP importance vector in partitions of for synergy,
redundancy, and independence.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Iterable, List, Optional, Tuple, TypeVar

import numpy as np

from pytools.api import AllTracker, inheritdoc

from ._shap import ShapCalculator
from ._shap_global_explanation import (
    AffinityMatrix,
    ShapContext,
    ShapGlobalExplainer,
    ShapInteractionGlobalExplainer,
    ShapInteractionValueContext,
    ShapValueContext,
    cov,
    cov_broadcast,
    ensure_last_axis_is_fast,
    fill_diagonal,
    sqrt,
    transpose,
)

log = logging.getLogger(__name__)

__all__ = ["ShapInteractionVectorProjector", "ShapProjector", "ShapVectorProjector"]

#: if ``True``, optimize numpy arrays to ensure pairwise partial summation.
#: But given that we will add floats of the same order of magnitude and only up
#: to a few thousand of them in the base case, the loss of accuracy with regular
#: (sequential) summation will be negligible in practice
_PAIRWISE_PARTIAL_SUMMATION = False

#
# Type variables
#

T_Self = TypeVar("T_Self")

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class ShapProjector(ShapGlobalExplainer, metaclass=ABCMeta):
    """
    Base class for global pairwise model explanations based on SHAP vector projection.
    """

    def __init__(self) -> None:
        super().__init__()
        self.association_: Optional[AffinityMatrix] = None

    def association(
        self, absolute: bool, symmetrical: bool, std: bool = False
    ) -> Optional[np.ndarray]:
        """[see superclass]"""
        self._ensure_fitted()
        return self.association_.get_values(
            symmetrical=symmetrical, absolute=absolute, std=std
        )

    def _fit(self, shap_calculator: ShapCalculator) -> None:
        self._reset_fit()
        self._calculate(self._get_context(shap_calculator=shap_calculator))

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.association_ = None

    @abstractmethod
    def _get_context(self, shap_calculator: ShapCalculator) -> List[ShapContext]:
        pass

    @abstractmethod
    def _calculate(self, contexts: Iterable[ShapContext]) -> AffinityMatrix:
        pass

    @staticmethod
    def _calculate_association(context: ShapContext) -> AffinityMatrix:
        # Calculate association: ass[i, j]
        #
        # Input: shap context
        # returns: ass[i, j], with shape (n_outputs, n_features, n_features)

        cov_p_i_p_j = context.cov_p_i_p_j
        var_p_i = context.var_p_i

        # cov(p[i], p[j]) / var(p[i])
        # orthogonal projection of p[i, j] onto p[i]
        # this converges towards 0 as var(p[i]) converges towards 0
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j_over_var_p_i = np.zeros(cov_p_i_p_j.shape)
        np.divide(
            cov_p_i_p_j, var_p_i, out=cov_p_i_p_j_over_var_p_i, where=var_p_i > 0.0
        )

        # calculate association as the coefficient of determination for p[i] and p[j]
        ass_ij = cov_p_i_p_j_over_var_p_i * transpose(cov_p_i_p_j_over_var_p_i)

        # we define the association of a feature with itself as 1
        fill_diagonal(ass_ij, 1.0)

        return AffinityMatrix.from_relative_affinity(
            affinity_rel_ij=ass_ij, std_p_i=sqrt(var_p_i)
        )


class ShapVectorProjector(ShapProjector):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible pairings
    of features into additive components for association and independence.
    This is achieved through scalar projection of redundancy and synergy vectors
    onto a feature's main SHAP vector.
    """

    def _get_context(self, shap_calculator: ShapCalculator) -> List[ShapContext]:
        return [
            ShapValueContext(shap_calculator=shap_calculator, split_id=split_id)
            for split_id in range(shap_calculator.n_splits_)
        ]

    def _calculate(self, contexts: Iterable[ShapContext]) -> None:
        # calculate association matrices for each SHAP context, then aggregate
        self.association_ = AffinityMatrix.aggregate(
            affinity_matrices=map(self._calculate_association, contexts)
        )


@inheritdoc(match="""[see superclass]""")
class ShapInteractionVectorProjector(ShapProjector, ShapInteractionGlobalExplainer):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible pairings
    of features into additive components for synergy, redundancy, and independence.
    This is achieved through scalar projection of redundancy and synergy vectors
    onto a feature's main SHAP vector.
    SHAP interaction scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    def __init__(self) -> None:
        super().__init__()

        self.synergy_: Optional[AffinityMatrix] = None
        self.redundancy_: Optional[AffinityMatrix] = None

    def synergy(
        self, symmetrical: bool, absolute: bool, std: bool = False
    ) -> Optional[np.ndarray]:
        """[see superclass]"""
        self._ensure_fitted()
        return self.synergy_.get_values(
            symmetrical=symmetrical, absolute=absolute, std=std
        )

    def redundancy(
        self, symmetrical: bool, absolute: bool, std: bool = False
    ) -> Optional[np.ndarray]:
        """[see superclass]"""
        self._ensure_fitted()
        return self.redundancy_.get_values(
            symmetrical=symmetrical, absolute=absolute, std=std
        )

    def _get_context(self, shap_calculator: ShapCalculator) -> List[ShapContext]:
        return [
            ShapInteractionValueContext(
                shap_calculator=shap_calculator, split_id=split_id
            )
            for split_id in range(shap_calculator.n_splits_)
        ]

    def _calculate(self, contexts: Iterable[ShapContext]) -> None:
        # calculate association, synergy, and redundancy matrices for each SHAP context,
        # then aggregate each of them
        self.association_, self.synergy_, self.redundancy_ = map(
            AffinityMatrix.aggregate,
            zip(
                *(
                    (
                        self._calculate_association(context=context),
                        *self._calculate_synergy_redundancy(context=context),
                    )
                    for context in contexts
                )
            ),
        )

    def _calculate_synergy_redundancy(
        self, context: ShapContext
    ) -> Tuple[AffinityMatrix, AffinityMatrix]:
        p_i = context.p_i
        var_p_i = context.var_p_i
        p_ij = context.p_ij
        weight = context.weight

        #
        # Synergy: syn[i, j]
        #

        # var(p[i, j])
        # shape: (n_outputs, n_features, n_features)
        # variance of each feature interaction vector
        var_p_ij = np.average(
            ensure_last_axis_is_fast(p_ij ** 2), axis=-1, weights=weight
        )

        # cov(p[i], p[i, j])
        # covariance matrix of shap vectors with pairwise synergies
        # shape: (n_outputs, n_features, n_features)

        cov_p_i_p_ij = cov_broadcast(p_i, p_ij, weight=weight)
        matrix_shape = cov_p_i_p_ij.shape

        # cov(p[i], p[i, j]) / var(p[i, j])
        # orthogonal projection of p[i] onto p[i, j]
        # this converges towards 0 as var(p[i, j]) converges towards 0
        # shape: (n_outputs, n_features, n_features)

        cov_p_i_p_ij_over_var_p_ij = np.zeros(matrix_shape)
        np.divide(
            cov_p_i_p_ij, var_p_ij, out=cov_p_i_p_ij_over_var_p_ij, where=var_p_ij > 0.0
        )

        # cov(p[i], p[i, j]) / var(p[i])
        # orthogonal projection of p[i, j] onto p[i]
        # this converges towards 0 as var(p[i]) converges towards 0
        # shape: (n_outputs, n_features, n_features)

        cov_p_i_p_ij_over_var_p_i = np.zeros(matrix_shape)
        np.divide(
            cov_p_i_p_ij, var_p_i, out=cov_p_i_p_ij_over_var_p_i, where=var_p_i > 0.0
        )

        # syn[i, j]
        # this is the coefficient of determination of the interaction vector
        syn_ij = cov_p_i_p_ij_over_var_p_i * cov_p_i_p_ij_over_var_p_ij

        # we define the synergy of a feature with itself as 1
        fill_diagonal(syn_ij, 1.0)

        #
        # Redundancy: red[i, j]
        #

        # cov(p[i], p[j])
        # covariance matrix of shap vectors
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j = cov(p_i, weight)

        # nominator
        # shape: (n_outputs, n_features, n_features)
        red_ij_nominator = (
            cov_p_i_p_j - cov_p_i_p_ij_over_var_p_ij * transpose(cov_p_i_p_ij)
        ) ** 2

        # denominator for p_i
        # shape: (n_outputs, n_features, n_features)
        red_ij_denominator_i = var_p_i - cov_p_i_p_ij_over_var_p_ij * cov_p_i_p_ij
        red_ij_denominator = red_ij_denominator_i * transpose(red_ij_denominator_i)

        # red[i, j]
        # this converges towards 0 as the denominator converges towards 0
        # shape: (n_outputs, n_features, n_features)

        red_ij = np.zeros(matrix_shape)
        np.divide(
            red_ij_nominator,
            red_ij_denominator,
            out=red_ij,
            where=red_ij_denominator > 0.0,
        )

        # scale to accommodate variance already explained by synergy
        red_ij *= 1 - syn_ij

        # we define the redundancy of a feature with itself as 1
        fill_diagonal(red_ij, 1.0)

        #
        # SHAP decomposition as relative contributions of
        # synergy, redundancy, and independence
        #

        # we should have the right shape for all resulting matrices
        assert syn_ij.shape == matrix_shape
        assert red_ij.shape == matrix_shape

        # Calculate relative synergy and redundancy (ranging from 0.0 to 1.0),
        # as a symmetric and an asymmetric measure.
        # For the symmetric case, we ensure perfect symmetry by removing potential
        # round-off errors
        # NOTE: we do not store independence, so technically it could be removed from
        # the code above

        std_p_i = sqrt(var_p_i)
        return (
            AffinityMatrix.from_relative_affinity(
                affinity_rel_ij=syn_ij, std_p_i=std_p_i
            ),
            AffinityMatrix.from_relative_affinity(
                affinity_rel_ij=red_ij, std_p_i=std_p_i
            ),
        )

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.redundancy_ = None
        self.synergy_ = None


__tracker.validate()
