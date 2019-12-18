"""
Decomposition of SHAP contribution scores (i.e, SHAP importance) of all possible parings
of features into additive components for synergy, equivalence, and independence.
"""
import logging
from typing import *

import numpy as np
import pandas as pd

from gamma.common.fit import FittableMixin
from gamma.ml.inspection._shap import ShapInteractionValuesCalculator

log = logging.getLogger(__name__)


class ShapInteractionDecomposer(FittableMixin[ShapInteractionValuesCalculator]):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible parings
    of features into additive components for synergy, equivalence, and independence.
    SHAP interaction scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    DEFAULT_MIN_DIRECT_SYNERGY = 0.005

    def __init__(self, min_direct_synergy: Optional[float] = None) -> None:
        """
        :param min_direct_synergy: minimum direct synergy a pair of features \
            :math:`f_i' and :math:`f_j' needs to manifest in order to be considered \
            for calculating indirect synergies. This is expressed as the relative \
            contribution score with regard to the total synergistic contributions \
            ranging between 0 and 1, and calculated as \
            :math:`\\frac \
                    {\\sigma_{\\vec{\\phi_{ij}}}} \
                    {\\sum_{i,j}\\sigma_{\\vec{\\phi_{ij}}}}`, \
            i.e, the relative share of the synergy contribution \
            :math:`\\sigma_{\\vec{\\phi_{ij}}}`. \
        """
        f"""\
            (default: {ShapInteractionDecomposer.DEFAULT_MIN_DIRECT_SYNERGY}, \
            i.e., {ShapInteractionDecomposer.DEFAULT_MIN_DIRECT_SYNERGY * 100.0:g}%)
        """
        super().__init__()
        self.min_direct_synergy = (
            ShapInteractionDecomposer.DEFAULT_MIN_DIRECT_SYNERGY
            if min_direct_synergy is None
            else min_direct_synergy
        )
        self.synergy_rel_: Optional[np.ndarray] = None
        self.equivalence_rel_: Optional[np.ndarray] = None
        self.index_: Optional[pd.Index] = None
        self.columns_: Optional[pd.Index] = None

    @property
    def is_fitted(self) -> bool:
        f"""{FittableMixin.is_fitted.__doc__}"""
        return self.synergy_rel_ is not None

    @property
    def synergy(self) -> pd.DataFrame:
        """
        The matrix of total relative synergy (direct and indirect) for all feature
        pairs.

        Values range between 0.0 (fully autonomous contributions) and 1.0
        (fully synergistic contributions).

        Raises an error if this interaction decomposer has not been fitted.
        """
        self._ensure_fitted()
        return self._to_frame(self.synergy_rel_)

    @property
    def equivalence(self) -> pd.DataFrame:
        """
        The matrix of total relative equivalence for all feature pairs.

        Values range between 0.0 (fully unique contributions) and 1.0
        (fully equivalent contributions).

        Raises an error if this interaction decomposer has not been fitted.
        """
        self._ensure_fitted()
        return self._to_frame(self.equivalence_rel_)

    def fit(self, im_calculator: ShapInteractionValuesCalculator, **fit_params) -> None:
        """
        Calculate the SHAP decomposition for the interaction matrix produced by the
        given interaction matrix calculator.
        :param im_calculator: the calculator from which to get the interaction matrix
        """

        # reset fit in case we get an exception along the way
        self.synergy_rel_ = self.equivalence_rel_ = None
        self.index_ = self.columns_ = None

        if len(fit_params) > 0:
            raise ValueError(
                f'unsupported fit parameters: {", ".join(fit_params.values())}'
            )

        #
        # basic definitions
        #

        im: pd.DataFrame = im_calculator.matrix
        n_features = im_calculator.n_features_
        n_targets = im_calculator.n_targets_
        n_observations = im_calculator.n_observations_
        if n_targets == 1:
            targets: Optional[pd.Index] = None
            features: pd.Index = im.columns
        else:
            targets, features = im.columns.levels

        def _cov(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            # calculate covariance matrix of two vectors, assuming µ=0 for both
            # input shape for u and v is (n_targets, n_features)
            # output shape is (n_targets, n_features, n_features)

            assert u.shape == (n_targets, n_features, n_observations)
            assert v.shape == (n_targets, n_features, n_observations)

            return np.matmul(u, v.swapaxes(1, 2)) / n_observations

        def _transpose(v: np.ndarray) -> np.ndarray:
            # transpose a feature matrix for all targets
            assert v.shape == (n_targets, n_features, n_features)

            return v.swapaxes(1, 2)

        def _sqrt(v: np.ndarray) -> np.ndarray:
            # we clip values < 0 as these could happen in isolated cases due to
            # rounding errors

            return np.sqrt(np.clip(v, 0, None))

        # phi[i, j]
        # shape: (n_targets, n_features, n_features, n_observations)
        # the vector of interaction values for every target and feature pairing
        phi_ij = np.transpose(
            im.values.reshape((n_observations, n_features, n_targets, n_features)),
            axes=(2, 1, 3, 0),
        )

        # phi[i]
        # shape: (n_targets, n_features, n_observations)
        phi_i = phi_ij.sum(axis=2)

        # covariance matrix of shap vectors
        # shape: (n_targets, n_features, n_features)
        cov_phi_i_phi_j = _cov(phi_i, phi_i)

        #
        # Feature synergy (direct and indirect)
        #

        # var(phi[i, j]), std(phi[i, j])
        # shape: (n_targets, n_features, n_features)
        # variance and length (= standard deviation) of each feature interaction vector
        var_phi_ij = (phi_ij * phi_ij).mean(axis=3)
        std_phi_ij = np.sqrt(var_phi_ij)

        # phi[i, i]
        # shape: (n_targets, n_features, n_observations)
        # independent feature contributions;
        # this is the diagonal of phi_ij
        phi_ii = np.diagonal(phi_ij, axis1=1, axis2=2).swapaxes(1, 2)

        # phi'[i] = phi[i] - phi[i, i]
        # shape: (n_targets, n_features, n_observations)
        # the SHAP vectors per feature, minus the independent contributions
        phi__i = phi_i - phi_ii

        # std_phi_relative[i, j]
        # shape: (n_targets, n_features, n_features)
        # relative importance of phi[i, j] measured as the length of phi[i, j]
        # as percentage of the sum of lengths of all phi[..., ...]
        std_phi_relative_ij = std_phi_ij / std_phi_ij.sum()

        # phi_valid[i, j]
        # shape: (n_targets, n_features, n_features)
        # boolean matrix indicating whether phi[i, j] is above the "noise" threshold,
        # i.e. whether we trust that doing calculations with phi[i, j] is sufficiently
        # accurate. phi[i, j] with small variances should not be used because we divide
        # by that variance to determine the multiple for indirect synergy, which can
        # deviate significantly if var(phi[i, j]) is only slightly off

        interaction_noise_threshold = self.min_direct_synergy
        phi_valid_ij = std_phi_relative_ij >= interaction_noise_threshold

        # s[i, j]
        # shape: (n_targets, n_features, n_features)
        # s[i, j] = cov(phi'[i], phi[i, j) / var(phi[i, j]), for each target
        # this is the orthogonal projection of phi[i, j] onto phi'[i] and determines
        # the multiplier of std(phi[i, j]) (i.e., direct synergy) to obtain
        # the total direct and indirect synergy of phi'[i] with phi'[j], i.e.,
        # std(phi[i, j]) * s[i, j]

        # noinspection SpellCheckingInspection
        s_i_j = np.divide(
            # cov(phi'[i], phi[i, j])
            np.einsum("tio,tijo->tij", phi__i, phi_ij) / n_observations,
            # var(phi[i, j])
            var_phi_ij,
            out=np.ones_like(var_phi_ij),
            where=phi_valid_ij,
        )

        # issue warning messages for edge cases

        def _for_target(_t: int) -> str:
            if targets is None:
                return ""
            else:
                return f'for target "{targets[_t]}" and '

        def _relative_direct_synergy(_t: int, _i: int, _j: int) -> str:
            return (
                f"phi[{features[_i]}, {features[_j]}] has "
                f"{std_phi_relative_ij[_t, _i, _j] * 100:.3g}% "
                "relative SHAP contribution; "
                "consider increasing the minimal direct synergy threshold (currently "
                f"{interaction_noise_threshold * 100:.3g}%). "
            )

        def _test_synergy_feasibility() -> None:
            for _t, _i, _j in np.argwhere(s_i_j < 1):
                if _i != _j:
                    log.warning(
                        f"contravariant indirect synergy {_for_target(_t)}"
                        f"phi[{features[_i]}, {features[_j]}]: "
                        "indirect synergy calculated as "
                        f"{(s_i_j[_t, _i, _j] - 1) * 100:.3g}% "
                        "of direct synergy; setting indirect synergy to 0. "
                        f"{_relative_direct_synergy(_t,_i, _j)}"
                    )

            for _t, _i, _j in np.argwhere(s_i_j > np.log2(n_features)):
                if _i != _j:
                    log.warning(
                        f"high indirect synergy {_for_target(_t)}"
                        f"phi[{features[_i]}, {features[_j]}]: "
                        "total of direct and indirect synergy is "
                        f"{(s_i_j[_t, _i, _j] - 1) * 100:.3g}% of direct synergy. "
                        f"{_relative_direct_synergy(_t,_i, _j)}"
                    )

        _test_synergy_feasibility()

        # ensure that s[i, j] is at least 1.0
        # (a warning will have been issued during the checks above for s[i, j] < 1)
        # i.e. we don't allow total synergy to be less than direct synergy
        s_i_j = np.clip(s_i_j, 1.0, None)

        # fill the diagonal(s) with nan since these values are meaningless
        for s_i_j_for_target in s_i_j:
            np.fill_diagonal(s_i_j_for_target, val=np.nan)

        # s[j, i]
        # transpose of s[i, j]; we need this later for calculating SHAP equivalence
        # shape: (n_targets, n_features, n_features)
        s_j_i = _transpose(s_i_j)

        # zeta[i, j] = zeta[j, i] = (s[i, j] + s[j, i]) * phi[i, j]
        # total SHAP synergy, comprising both direct and indirect synergy
        # shape: (n_targets, n_features, n_features)
        std_zeta_ij_plus_zeta_ji = (s_i_j + s_j_i) * std_phi_ij

        #
        # SHAP independence
        #

        # cov(phi[i], phi[i, j])
        # covariance matrix of shap vectors with pairwise synergies
        # shape: (n_targets, n_features, n_features)
        cov_phi_i_phi_ij = (
            np.einsum("...io,...ijo->...ij", phi_i, phi_ij) / n_observations
        )

        # cov(psi[i, j], psi[j, i])
        # where psi[i, j] = phi[i] - s[i, j] * phi[i, j]
        # shape: (n_observations, n_targets, n_features)
        # matrix of covariances for psi vectors for all pairings of features
        # the psi[i, j] vector is the phi[i] SHAP contribution vector where the synergy
        # effects (direct and indirect) with feature j have been deducted

        cov_psi_ij_psi_ji = (
            cov_phi_i_phi_j
            - s_j_i * cov_phi_i_phi_ij
            - s_i_j * _transpose(cov_phi_i_phi_ij)
            + s_j_i * s_i_j * var_phi_ij
        )

        # var(phi[i])
        # variances of SHAP vectors
        # shape: (n_targets, n_features, 1)
        # i.e. adding a second, empty feature dimension to enable correct broadcasting
        var_phi_ix = np.diagonal(cov_phi_i_phi_j, axis1=1, axis2=2)[:, :, np.newaxis]

        # var(psi[i, j])
        # variances of SHAP vectors minus total synergy
        # shape: (n_targets, n_features, n_features)
        var_psi_ij = (
            var_phi_ix - 2 * s_i_j * cov_phi_i_phi_ij + s_i_j * s_i_j * var_phi_ij
        )

        # var(phi[i]) + var(phi_i[j])
        # Sum of covariances per feature pair (this is a diagonal matrix)
        # shape: (n_targets, n_features, n_features)
        var_psi_ij_plus_var_psi_ji = var_psi_ij + _transpose(var_psi_ij)

        # 2 * cov(psi[i, j], psi[j, i])
        # shape: (n_targets, n_features, n_features)
        # this is an intermediate result to calculate the standard deviation
        # of the equivalence vector (see next step below)
        cov_psi_ij_psi_ji_2x = 2 * cov_psi_ij_psi_ji

        # std(psi[i, j] + psi[j, i])
        # where psi[i, j] = phi[i] - zeta[i, j]
        # shape: (n_targets, n_features, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors psi[i, j] and psi[j, i]
        # this quantifies the autonomous contributions of features i and j, i.e.,
        # without synergizing
        # we also need this as part of the formula to calculate epsilon_ij (see below)

        std_psi_ij_plus_psi_ji = _sqrt(
            var_psi_ij_plus_var_psi_ji + cov_psi_ij_psi_ji_2x
        )

        #
        # SHAP equivalence
        #

        # std(psi[i, j] - psi[j, i])
        # shape: (n_targets, n_features, n_features)
        # the length of the difference of vectors psi[i, j] and psi[j, i]
        # we need this as part of the formula to calculate epsilon_ij (see below)

        std_psi_ij_minus_psi_ji = _sqrt(
            var_psi_ij_plus_var_psi_ji - cov_psi_ij_psi_ji_2x
        )

        # 2 * std(epsilon[i, j]) = 2 * std(epsilon[j, i])
        # shape: (n_targets, n_features)
        # twice the standard deviation (= length) of equivalence vector;
        # this is the total contribution made by phi[i] and phi[j] where both features
        # independently use redundant information

        std_epsilon_ij_2x = std_psi_ij_plus_psi_ji - std_psi_ij_minus_psi_ji

        #
        # SHAP independence
        #

        # 4 * var(epsilon[i, j]), var(epsilon[i, j])
        # shape: (n_targets, n_features, n_features)
        var_epsilon_ij_4x = std_epsilon_ij_2x * std_epsilon_ij_2x

        # ratio of length of 2*e over length of (psi[i, j] + psi[j, i])
        # shape: (n_targets, n_features, n_features)
        # we need this for the next step
        epsilon_psi_ratio_2x = 1 - std_psi_ij_minus_psi_ji / std_psi_ij_plus_psi_ji

        # 2 * cov(psi[i, j], epsilon[i, j])
        # shape: (n_targets, n_features, n_features)
        # we need this as part of the formula to calculate tau_i (see next step below)
        cov_psi_ij_epsilon_ij_2x = epsilon_psi_ratio_2x * (
            var_psi_ij + cov_psi_ij_psi_ji
        )

        # std(tau_ij)
        # where tau_ij = psi_ij + psi_ji - 2 * epsilon_ij
        # shape: (n_targets, n_features, n_features)
        # the standard deviation (= length) of the independence vector

        std_tau_ij_plus_tau_ji = _sqrt(
            var_psi_ij
            + _transpose(var_psi_ij)
            + var_epsilon_ij_4x
            + 2
            * (
                cov_psi_ij_psi_ji
                - cov_psi_ij_epsilon_ij_2x
                - _transpose(cov_psi_ij_epsilon_ij_2x)
            )
        )

        #
        # SHAP uniqueness
        #

        # 2 * cov(phi[i], epsilon[i, j])
        # shape: (n_targets, n_features, n_features)
        # intermediate result to calculate upsilon[i, j], see next step

        cov_phi_i_epsilon_ij_2x = epsilon_psi_ratio_2x * (
            var_phi_ix + cov_phi_i_phi_j - (s_i_j + s_j_i) * cov_phi_i_phi_ij
        )

        # std(upsilon[i, j])
        # where upsilon[i, j] = phi[i] + phi[j] - 2 * epsilon[i, j]
        # shape: (n_targets, n_features, n_features)
        # this is the sum of complementary contributions of feature i and feature j,
        # i.e., deducting the equivalent contributions

        std_upsilon_ij_plus_upsilon_ji = _sqrt(
            var_phi_ix
            + var_phi_ix.swapaxes(1, 2)
            + var_epsilon_ij_4x
            + 2
            * (
                cov_phi_i_phi_j
                - cov_phi_i_epsilon_ij_2x
                - _transpose(cov_phi_i_epsilon_ij_2x)
            )
        )

        #
        # SHAP decompositon as relative contributions of
        # synergy, equivalence, and independence
        #

        synergy_ij = std_zeta_ij_plus_zeta_ji
        autonomy_ij = std_psi_ij_plus_psi_ji
        equivalence_ij = np.abs(std_epsilon_ij_2x)
        uniqueness_ij = std_upsilon_ij_plus_upsilon_ji
        independence_ij = std_tau_ij_plus_tau_ji

        # we should have the right shape for all resulting matrices
        for matrix in (
            synergy_ij,
            equivalence_ij,
            autonomy_ij,
            uniqueness_ij,
            independence_ij,
        ):
            assert matrix.shape == (n_targets, n_features, n_features)

        # assign results as an atomic operation, so we don't have a semi-fitted
        # outcome in case an exception is raised in this final step
        # NOTE: we do not store independence so technically it could be removed from
        # the code above

        self.synergy_rel_, self.equivalence_rel_ = (
            synergy_ij / (synergy_ij + autonomy_ij),
            equivalence_ij / (equivalence_ij + uniqueness_ij),
        )
        self.index_ = features
        self.columns_ = im.columns

    def _to_frame(self, matrix: np.ndarray) -> pd.DataFrame:
        # takes an array of shape (n_targets, n_features, n_features) and transforms it
        # into a data frame of shape (n_features, n_targets * n_features)
        index = self.index_
        columns = self.columns_
        return pd.DataFrame(
            matrix.swapaxes(0, 1).reshape(len(index), len(columns)),
            index=index,
            columns=columns,
        )
