"""
Projection of SHAP contribution scores (i.e, SHAP importance) of all possible
pairings of features onto the SHAP importance vector in partitions of for synergy,
redundancy, and independence.
"""
import logging
from typing import List, Optional, TypeVar

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc

from ._shap import ShapCalculator, ShapInteractionValuesCalculator
from ._shap_global_explanation import (
    AffinityMatrices,
    ShapGlobalExplainer,
    ShapInteractionGlobalExplainer,
    cov,
    cov_broadcast,
    diagonal,
    ensure_last_axis_is_fast,
    fill_diagonal,
    sqrt,
    transpose,
)

log = logging.getLogger(__name__)

__all__ = [
    "ShapInteractionProjector",
    "ShapProjector",
]

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
class ShapProjector(ShapGlobalExplainer):
    """
    Decomposes SHAP vectors (i.e., SHAP contribution) of all possible pairings
    of features into additive components for association and independence.
    SHAP contribution scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.association_: Optional[AffinityMatrices] = None

    def association(self, absolute: bool, symmetrical: bool) -> np.ndarray:
        """[see superclass]"""
        self._ensure_fitted()
        return self.association_.get_matrix(symmetrical=symmetrical, absolute=absolute)

    def _fit(self, shap_calculator: ShapCalculator) -> None:
        #
        # basic definitions
        #

        shap_values: pd.DataFrame = shap_calculator.get_shap_values(consolidate="mean")
        n_outputs: int = len(shap_calculator.output_names_)
        n_features: int = len(shap_calculator.feature_index_)
        n_observations: int = len(shap_values)

        # weights
        # shape: (n_observations)
        # return a 1d array of weights that aligns with the observations axis of the
        # SHAP values tensor (axis 1)
        weight: Optional[np.ndarray]

        _weight_sr = shap_calculator.sample_.weight
        if _weight_sr is not None:
            weight = _weight_sr.loc[shap_values.index].values
        else:
            weight = None

        # p[i] = p_i
        # shape: (n_outputs, n_features, n_observations)
        # the vector of shap values for every output and feature
        p_i = ensure_last_axis_is_fast(
            np.transpose(
                shap_values.values.reshape((n_observations, n_outputs, n_features)),
                axes=(1, 2, 0),
            )
        )

        # covariance matrix of shap vectors
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j = cov(p_i, weight)

        # variance of shap vectors
        # (n_outputs, n_features, 1)
        var_p_i = diagonal(cov_p_i_p_j)[:, :, np.newaxis]

        # cov(p[i], p[j]) / var(p[j])
        # orthogonal projection of p[i] onto p[j]
        # this converges towards 0 as var(p[j]) converges towards 0
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j_over_var_p_i = np.zeros((n_outputs, n_features, n_features))
        np.divide(
            cov_p_i_p_j, var_p_i, out=cov_p_i_p_j_over_var_p_i, where=var_p_i > 0.0
        )

        #
        # Association: ass[i, j]
        #

        # ass[i, j] = cov(p[i], p[j]) ** 2 / (var(p[i]) * var(p[j]))
        association_rel_ij = cov_p_i_p_j_over_var_p_i * transpose(
            cov_p_i_p_j_over_var_p_i
        )

        # we define the synergy of a feature with itself as 0
        fill_diagonal(association_rel_ij, 0)

        # we should have the right shape for all resulting matrices
        assert association_rel_ij.shape == (n_outputs, n_features, n_features)

        self.association_ = AffinityMatrices(
            affinity_rel_ij=association_rel_ij, std_p_i=sqrt(var_p_i)
        )

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.association_ = None


@inheritdoc(match="""[see superclass]""")
class ShapInteractionProjector(ShapProjector, ShapInteractionGlobalExplainer):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible pairings
    of features into additive components for synergy, redundancy, and independence.
    This is achieved through orthogonal projection of synergy and redundancy vectors
    onto a feature's main SHAP vector.
    SHAP interaction scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    #: if ``True``, orthogonalize SHAP interaction vectors before calculating SHAP
    #: projections
    orthogonalize: bool

    def __init__(self, orthogonalize: bool = True) -> None:
        super().__init__()

        self.orthogonalize = orthogonalize

        self.synergy_: Optional[AffinityMatrices] = None
        self.redundancy_: Optional[AffinityMatrices] = None

    def synergy(self, symmetrical: bool, absolute: bool) -> np.ndarray:
        """[see superclass]"""
        self._ensure_fitted()
        return self.synergy_.get_matrix(symmetrical=symmetrical, absolute=absolute)

    def redundancy(self, symmetrical: bool, absolute: bool) -> np.ndarray:
        """[see superclass]"""
        self._ensure_fitted()
        return self.redundancy_.get_matrix(symmetrical=symmetrical, absolute=absolute)

    @staticmethod
    def _get_orthogonalized_interaction_vectors(
        p_ij: np.ndarray, weight: Optional[np.ndarray]
    ) -> np.ndarray:
        # p_ij: shape: (n_outputs, n_features, n_features, n_observations)

        assert p_ij.ndim == 4
        n_features = p_ij.shape[1]
        assert p_ij.shape[2] == n_features

        # p[i, i]
        # shape: (n_outputs, n_features, n_observations)
        # independent feature contributions;
        # this is the diagonal of p[i, j], i.e., the main effects p[i, i]
        p_ii = p_ij.diagonal(axis1=1, axis2=2).swapaxes(1, 2)

        # cov[p[i, i], p[j, j]]
        # shape: (n_outputs, n_features, n_features)
        # covariance matrix of the main effects p[i, i]
        cov_p_ii_p_jj = cov(p_ii, weight=weight)

        # var[p[i, i]]
        # shape: (n_outputs, n_features, 1)
        # variance of the main effects p[i, i] as a broadcastable matrix where each
        # column is identical
        var_p_ii = diagonal(cov_p_ii_p_jj)[:, :, np.newaxis]

        # var[p[j, j]]
        # shape: (n_outputs, 1, n_features)
        # variance of the main effects p[j, j] as a broadcastable matrix where each
        # row is identical
        var_p_jj = transpose(var_p_ii)

        # cov[p[i, i], p[i, j]]
        # shape: (n_outputs, n_features, n_features)
        # covariance matrix of the main effects p[i, i] with interaction effects p[i, j]
        cov_p_ii_p_ij = cov_broadcast(p_ii, p_ij, weight=weight)

        # adjustment_factors[i, j]
        # shape: (n_outputs, n_features, n_features)
        # multiple of p[i, i] to be subtracted from p[i, j] and added to p[i, i]
        # to orthogonalize the SHAP interaction vectors

        _nominator = cov_p_ii_p_jj * transpose(cov_p_ii_p_ij) - cov_p_ii_p_ij * var_p_jj
        fill_diagonal(_nominator, 0.0)

        _denominator = cov_p_ii_p_jj ** 2 - var_p_ii * var_p_jj

        # The denominator is <= 0 due to the Cauchy-Schwarz inequality.
        # It is 0 only if the variance of p_ii or p_jj are zero (i.e., no main effect).
        # In that fringe case, the nominator will also be zero and we set the adjustment
        # factor to 0 (intuitively, there is nothing to adjust in a zero-length vector)
        adjustment_factors_ij = np.zeros(_nominator.shape)
        # todo: prevent catastrophic cancellation where nominator/denominator are ~0.0
        np.divide(
            _nominator,
            _denominator,
            out=adjustment_factors_ij,
            where=_denominator < 0.0,
        )

        fill_diagonal(adjustment_factors_ij, -adjustment_factors_ij.sum(axis=-1))

        delta_ij = (
            adjustment_factors_ij[:, :, :, np.newaxis] * p_ii[:, :, np.newaxis, :]
        )
        delta_ji = transpose(delta_ij)

        return p_ij - delta_ij - delta_ji

    def _fit(self, shap_calculator: ShapInteractionValuesCalculator) -> None:
        super()._fit(shap_calculator)

        #
        # basic definitions
        #
        shap_values: pd.DataFrame = shap_calculator.get_shap_interaction_values(
            consolidate="mean"
        )
        features: pd.Index = shap_calculator.feature_index_
        outputs: List[str] = shap_calculator.output_names_
        n_features: int = len(features)
        n_outputs: int = len(outputs)
        n_observations: int = shap_values.shape[0] // n_features

        # weights
        # shape: (n_observations)
        # return a 1d array of weights that aligns with the observations axis of the
        # SHAP values tensor (axis 1)
        weight: Optional[np.ndarray]

        _weight_sr = shap_calculator.sample_.weight
        if _weight_sr is not None:
            _observation_indices = shap_values.index.get_level_values(0).values.reshape(
                (n_observations, n_features)
            )[:, 0]
            weight = ensure_last_axis_is_fast(
                _weight_sr.loc[_observation_indices].values
            )
        else:
            weight = None

        # p[i, j]
        # shape: (n_outputs, n_features, n_features, n_observations)
        # the vector of interaction values for every output and feature pairing
        # for improved numerical precision, we ensure the last axis is the fast axis
        # i.e. stride size equals item size (see documentation for numpy.sum)
        p_ij = ensure_last_axis_is_fast(
            np.transpose(
                shap_values.values.reshape(
                    (n_observations, n_features, n_outputs, n_features)
                ),
                axes=(2, 1, 3, 0),
            )
        )

        if self.orthogonalize:
            p_ij = ensure_last_axis_is_fast(
                self._get_orthogonalized_interaction_vectors(p_ij=p_ij, weight=weight)
            )

        # p[i]
        # shape: (n_outputs, n_features, n_observations)
        p_i = ensure_last_axis_is_fast(p_ij.sum(axis=2))

        # covariance matrix of shap vectors
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j = cov(p_i, weight)

        # var(p[i])
        # variances of SHAP vectors
        # shape: (n_outputs, n_features, 1)
        # i.e. adding a second, empty feature dimension to enable correct broadcasting
        var_p_i = diagonal(cov_p_i_p_j)[:, :, np.newaxis]

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

        # cov(p[i], p[i, j]) / var(p[i, j])
        # orthogonal projection of p[i] onto p[i, j]
        # this converges towards 0 as var(p[i, j]) converges towards 0
        # shape: (n_outputs, n_features, n_features)

        cov_p_i_p_ij_over_var_p_ij = np.zeros((n_outputs, n_features, n_features))
        np.divide(
            cov_p_i_p_ij, var_p_ij, out=cov_p_i_p_ij_over_var_p_ij, where=var_p_ij > 0.0
        )

        # cov(p[i], p[i, j]) / var(p[i])
        # orthogonal projection of p[i, j] onto p[i]
        # this converges towards 0 as var(p[i]) converges towards 0
        # shape: (n_outputs, n_features, n_features)

        cov_p_i_p_ij_over_var_p_i = np.zeros((n_outputs, n_features, n_features))
        np.divide(
            cov_p_i_p_ij, var_p_i, out=cov_p_i_p_ij_over_var_p_i, where=var_p_i > 0.0
        )

        #
        # Synergy: syn[i, j]
        #

        syn_ij = cov_p_i_p_ij_over_var_p_ij * cov_p_i_p_ij_over_var_p_i

        # we define the synergy of a feature with itself as 1
        fill_diagonal(syn_ij, 1.0)

        #
        # Redundancy: red[i, j]
        #

        _nominator = (
            cov_p_i_p_j - cov_p_i_p_ij_over_var_p_ij * transpose(cov_p_i_p_ij)
        ) ** 2

        _denominator_ij = cov_p_i_p_ij_over_var_p_ij * cov_p_i_p_ij - var_p_i
        _denominator = _denominator_ij * transpose(_denominator_ij)

        red_ij = np.zeros(_nominator.shape)
        # todo: prevent catastrophic cancellation where nominator/denominator are ~0.0
        np.divide(_nominator, _denominator, out=red_ij, where=_nominator > 0.0)
        red_ij *= 1 - syn_ij

        # we define the redundancy of a feature with itself as 1
        fill_diagonal(red_ij, 1.0)

        #
        # SHAP independence: ind[i, j]
        #

        ind_ij = np.ones((n_outputs, n_features, n_features)) - syn_ij - red_ij
        fill_diagonal(ind_ij, 1.0)

        #
        # SHAP decomposition as relative contributions of
        # synergy, redundancy, and independence
        #

        # we should have the right shape for all resulting matrices
        for matrix in (syn_ij, red_ij, ind_ij):
            assert matrix.shape == (n_outputs, n_features, n_features)

        # Calculate relative synergy and redundancy (ranging from 0.0 to 1.0),
        # as a symmetric and an asymmetric measure.
        # For the symmetric case, we ensure perfect symmetry by removing potential
        # round-off errors
        # NOTE: we do not store independence, so technically it could be removed from
        # the code above

        std_p_i = sqrt(var_p_i)
        self.synergy_ = AffinityMatrices(affinity_rel_ij=syn_ij, std_p_i=std_p_i)
        self.redundancy_ = AffinityMatrices(affinity_rel_ij=red_ij, std_p_i=std_p_i)

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.redundancy_ = None
        self.synergy_ = None


__tracker.validate()
