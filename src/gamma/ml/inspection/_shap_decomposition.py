"""
Decomposition of SHAP contribution scores (i.e, SHAP importance) of all possible parings
of features into additive components for synergy, equivalence, and independence.
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd

from gamma.common.fit import FittableMixin
from gamma.ml.inspection._shap import (
    BaseShapCalculator,
    ShapInteractionValuesCalculator,
    ShapValuesCalculator,
)

log = logging.getLogger(__name__)

T_ShapCalculator = TypeVar("T_ShapCalculator", bound=BaseShapCalculator)

_PAIRWISE_PARTIAL_SUMMATION = False
#: if `True`, optimize numpy arrays to ensure pairwise partial summation.
#: But given that we will add floats of the same order of magnitude and only up
#: to a few thousand of them in the base case, the loss of accuracy with regular
#: (sequential) summation will be negligible in practice


class BaseShapDecomposer(
    FittableMixin[T_ShapCalculator], ABC, Generic[T_ShapCalculator]
):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible parings
    of features into additive components and calculates a matrix of scores for all
    feature pairings.
    """

    def __init__(self) -> None:
        super().__init__()
        self.index_: Optional[pd.Index] = None
        self.columns_: Optional[pd.Index] = None

    @property
    def is_fitted(self) -> bool:
        """[inherit docstring from parent class]"""
        return self.index_ is not None

    is_fitted.__doc__ = FittableMixin.is_fitted.__doc__

    def fit(self, shap_calculator: BaseShapCalculator, **fit_params) -> None:
        """
        Calculate the SHAP decomposition for the shap values produced by the
        given interaction shap values calculator.
        :param shap_calculator: the fitted calculator from which to get the shap values
        """
        successful = False
        try:
            if len(fit_params) > 0:
                raise ValueError(
                    f'unsupported fit parameters: {", ".join(fit_params.values())}'
                )

            shap_values = shap_calculator.shap_values

            self._fit(
                shap_values=shap_values,
                features=shap_calculator.features_,
                targets=shap_calculator.targets_,
            )

            self.index_ = pd.Index(shap_calculator.features_)
            self.columns_ = shap_values.columns

            successful = True

        finally:
            # reset fit in case we get an exception along the way
            if not successful:
                self._reset_fit()

    @abstractmethod
    def _fit(
        self, shap_values: pd.DataFrame, features: List[str], targets: List[str]
    ) -> None:
        pass

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        self.index_ = self.columns_ = None

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


class ShapValueDecomposer(BaseShapDecomposer[ShapValuesCalculator]):
    """
    Decomposes SHAP vectors (i.e, SHAP importance) of all possible parings
    of features into additive components for association and independence.
    SHAP interaction scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.association_rel_: Optional[np.ndarray] = None

    @property
    def association(self) -> pd.DataFrame:
        """
        The matrix of relative association for all feature pairs.

        Values range between 0.0 (fully associated contributions) and 1.0
        (fully independent contributions).

        Raises an error if this SHAP value decomposer has not been fitted.
        """
        self._ensure_fitted()
        return self._to_frame(self.association_rel_)

    def _fit(
        self, shap_values: pd.DataFrame, features: List[str], targets: List[str]
    ) -> None:
        #
        # basic definitions
        #

        n_targets = len(targets)
        n_features = len(features)
        n_observations = len(shap_values)

        # phi[i]
        # shape: (n_targets, n_features, n_observations)
        # the vector of shap values for every target and feature
        phi_i = _ensure_last_axis_is_fast(
            np.transpose(
                shap_values.values.reshape((n_observations, n_targets, n_features)),
                axes=(1, 2, 0),
            )
        )

        #
        # SHAP association: alpha[i, j]
        #

        # covariance matrix of shap vectors
        # shape: (n_targets, n_features, n_features)
        cov_phi_i_phi_j = _cov(phi_i, phi_i)
        cov_phi_i_phi_j_2x = 2 * cov_phi_i_phi_j

        # variance of shap vectors
        var_phi_i = np.diagonal(cov_phi_i_phi_j, axis1=1, axis2=2)
        var_phi_i_plus_var_phi_j = (
            var_phi_i[:, :, np.newaxis] + var_phi_i[:, np.newaxis, :]
        )

        # std(phi[i] + phi[j])
        # shape: (n_targets, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors phi[i] and phi[j]
        # this quantifies the joint contributions of features i and j
        # we also need this as part of the formula to calculate alpha_ij (see below)
        std_phi_i_plus_phi_j = _sqrt(var_phi_i_plus_var_phi_j + cov_phi_i_phi_j_2x)

        # std(phi[i] - phi[j])
        # shape: (n_targets, n_features)
        # the length of the difference of vectors phi[i] and phi[j]
        # we need this as part of the formula to calculate alpha_ij (see below)
        std_phi_i_minus_phi_j = _sqrt(var_phi_i_plus_var_phi_j - cov_phi_i_phi_j_2x)

        # 2 * std(alpha[i, j]) = 2 * std(alpha[j, i])
        # shape: (n_targets, n_features, n_features)
        # twice the standard deviation (= length) of equivalence vector;
        # this is the total contribution made by phi[i] and phi[j] where both features
        # independently use redundant information
        std_alpha_ij_2x = std_phi_i_plus_phi_j - std_phi_i_minus_phi_j

        # SHAP association
        # shape: (n_targets, n_features, n_features)
        association_ij = np.abs(std_alpha_ij_2x)

        # we should have the right shape for all resulting matrices
        assert association_ij.shape == (n_targets, n_features, n_features)
        self.association_rel_ = association_ij / std_phi_i_plus_phi_j

    def _reset_fit(self) -> None:
        super()._reset_fit()
        self.association_rel_ = None


class ShapInteractionValueDecomposer(
    BaseShapDecomposer[ShapInteractionValuesCalculator]
):
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
        super().__init__()
        self.min_direct_synergy = (
            ShapInteractionValueDecomposer.DEFAULT_MIN_DIRECT_SYNERGY
            if min_direct_synergy is None
            else min_direct_synergy
        )
        self.synergy_rel_: Optional[np.ndarray] = None
        self.equivalence_rel_: Optional[np.ndarray] = None

    __init__.__doc__ += f"""\
            (default: {DEFAULT_MIN_DIRECT_SYNERGY}, i.e., \
            {DEFAULT_MIN_DIRECT_SYNERGY * 100.0:g}%)
        """

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

    def _fit(
        self, shap_values: pd.DataFrame, features: List[str], targets: List[str]
    ) -> None:
        #
        # basic definitions
        #

        n_features = len(features)
        n_targets = len(targets)
        n_observations = len(shap_values.index.levels[0])

        # phi[i, j]
        # shape: (n_targets, n_features, n_features, n_observations)
        # the vector of interaction values for every target and feature pairing
        # for improved numerical precision, we ensure the last axis is the fast axis
        # i.e. stride size equals item size (see documentation for numpy.sum)
        phi_ij = _ensure_last_axis_is_fast(
            np.transpose(
                shap_values.values.reshape(
                    (n_observations, n_features, n_targets, n_features)
                ),
                axes=(2, 1, 3, 0),
            )
        )

        # phi[i]
        # shape: (n_targets, n_features, n_observations)
        phi_i = _ensure_last_axis_is_fast(phi_ij.sum(axis=2))

        # covariance matrix of shap vectors
        # shape: (n_targets, n_features, n_features)
        cov_phi_i_phi_j = _cov(phi_i, phi_i)

        #
        # Feature synergy (direct and indirect): zeta[i, j]
        #

        # var(phi[i, j]), std(phi[i, j])
        # shape: (n_targets, n_features, n_features)
        # variance and length (= standard deviation) of each feature interaction vector
        var_phi_ij = (
            _ensure_last_axis_is_fast(phi_ij * phi_ij).sum(axis=-1) / n_observations
        )
        std_phi_ij = np.sqrt(var_phi_ij)

        # phi[i, i]
        # shape: (n_targets, n_features, n_observations)
        # independent feature contributions;
        # this is the diagonal of phi_ij
        phi_ii = np.diagonal(phi_ij, axis1=1, axis2=2).swapaxes(1, 2)

        # psi[i] = phi[i] - phi[i, i]
        # shape: (n_targets, n_features, n_observations)
        # the SHAP vectors per feature, minus the independent contributions
        psi_i = _ensure_last_axis_is_fast(phi_i - phi_ii)

        # std_phi_relative[i, j]
        # shape: (n_targets, n_features, n_features)
        # relative importance of phi[i, j] measured as the length of phi[i, j]
        # as percentage of the sum of lengths of all phi[..., ...]
        std_phi_relative_ij = std_phi_ij / std_phi_ij.sum()

        # phi_valid[i, j]
        # shape: (n_targets, n_features, n_features)
        # boolean values indicating whether phi[i, j] is above the "noise" threshold,
        # i.e. whether we trust that doing calculations with phi[i, j] is sufficiently
        # accurate. phi[i, j] with small variances should not be used because we divide
        # by that variance to determine the multiple for indirect synergy, which can
        # deviate significantly if var(phi[i, j]) is only slightly off

        interaction_noise_threshold = self.min_direct_synergy
        phi_valid_ij = std_phi_relative_ij >= interaction_noise_threshold

        # s[i, j]
        # shape: (n_targets, n_features, n_features)
        # s[i, j] = cov(psi[i], phi[i, j) / var(phi[i, j]), for each target
        # this is the orthogonal projection of phi[i, j] onto psi[i] and determines
        # the multiplier of std(phi[i, j]) (i.e., direct synergy) to obtain
        # the total direct and indirect synergy of psi[i] with psi[j], i.e.,
        # std(phi[i, j]) * s[i, j]

        if _PAIRWISE_PARTIAL_SUMMATION:
            raise NotImplementedError(
                "max precision Einstein summation not yet implemented"
            )
        # noinspection SpellCheckingInspection
        s_i_j = np.divide(
            # cov(psi[i], phi[i, j])
            np.einsum("tio,tijo->tij", psi_i, phi_ij, optimize=True) / n_observations,
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
        # SHAP autonomy: tau[i, j]
        #

        # cov(phi[i], phi[i, j])
        # covariance matrix of shap vectors with pairwise synergies
        # shape: (n_targets, n_features, n_features)

        if _PAIRWISE_PARTIAL_SUMMATION:
            raise NotImplementedError(
                "max precision Einstein summation not yet implemented"
            )
        cov_phi_i_phi_ij = (
            np.einsum("...io,...ijo->...ij", phi_i, phi_ij) / n_observations
        )

        # cov(tau[i, j], tau[j, i])
        # where tau[i, j] = phi[i] - s[i, j] * phi[i, j]
        # shape: (n_observations, n_targets, n_features)
        # matrix of covariances for tau vectors for all pairings of features
        # the tau[i, j] vector is the phi[i] SHAP contribution vector where the synergy
        # effects (direct and indirect) with feature j have been deducted

        cov_tau_ij_tau_ji = (
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

        # var(tau[i, j])
        # variances of SHAP vectors minus total synergy
        # shape: (n_targets, n_features, n_features)
        var_tau_ij = (
            var_phi_ix - 2 * s_i_j * cov_phi_i_phi_ij + s_i_j * s_i_j * var_phi_ij
        )

        # var(phi[i]) + var(phi_i[j])
        # Sum of covariances per feature pair (this is a diagonal matrix)
        # shape: (n_targets, n_features, n_features)
        var_tau_ij_plus_var_tau_ji = var_tau_ij + _transpose(var_tau_ij)

        # 2 * cov(tau[i, j], tau[j, i])
        # shape: (n_targets, n_features, n_features)
        # this is an intermediate result to calculate the standard deviation
        # of the equivalence vector (see next step below)
        cov_tau_ij_tau_ji_2x = 2 * cov_tau_ij_tau_ji

        # std(tau[i, j] + tau[j, i])
        # where tau[i, j] = phi[i] - zeta[i, j]
        # shape: (n_targets, n_features, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors tau[i, j] and tau[j, i]
        # this quantifies the autonomous contributions of features i and j, i.e.,
        # without synergizing
        # we also need this as part of the formula to calculate epsilon_ij (see below)

        std_tau_ij_plus_tau_ji = _sqrt(
            var_tau_ij_plus_var_tau_ji + cov_tau_ij_tau_ji_2x
        )

        #
        # SHAP equivalence: epsilon[i, j]
        #

        # std(tau[i, j] - tau[j, i])
        # shape: (n_targets, n_features, n_features)
        # the length of the difference of vectors tau[i, j] and tau[j, i]
        # we need this as part of the formula to calculate epsilon_ij (see below)

        std_tau_ij_minus_tau_ji = _sqrt(
            var_tau_ij_plus_var_tau_ji - cov_tau_ij_tau_ji_2x
        )

        # 2 * std(epsilon[i, j]) = 2 * std(epsilon[j, i])
        # shape: (n_targets, n_features)
        # twice the standard deviation (= length) of equivalence vector;
        # this is the total contribution made by phi[i] and phi[j] where both features
        # independently use redundant information

        std_epsilon_ij_2x = std_tau_ij_plus_tau_ji - std_tau_ij_minus_tau_ji

        #
        # SHAP independence: nu[i, j]
        #

        # 4 * var(epsilon[i, j]), var(epsilon[i, j])
        # shape: (n_targets, n_features, n_features)
        var_epsilon_ij_4x = std_epsilon_ij_2x * std_epsilon_ij_2x

        # ratio of length of 2*e over length of (tau[i, j] + tau[j, i])
        # shape: (n_targets, n_features, n_features)
        # we need this for the next step
        epsilon_tau_ratio_2x = 1 - std_tau_ij_minus_tau_ji / std_tau_ij_plus_tau_ji

        # 2 * cov(tau[i, j], epsilon[i, j])
        # shape: (n_targets, n_features, n_features)
        # we need this as part of the formula to calculate nu_i (see next step below)
        cov_tau_ij_epsilon_ij_2x = epsilon_tau_ratio_2x * (
            var_tau_ij + cov_tau_ij_tau_ji
        )

        # std(nu_ij)
        # where nu_ij = tau_ij + tau_ji - 2 * epsilon_ij
        # shape: (n_targets, n_features, n_features)
        # the standard deviation (= length) of the independence vector

        std_nu_ij_plus_nu_ji = _sqrt(
            var_tau_ij
            + _transpose(var_tau_ij)
            + var_epsilon_ij_4x
            + 2
            * (
                cov_tau_ij_tau_ji
                - cov_tau_ij_epsilon_ij_2x
                - _transpose(cov_tau_ij_epsilon_ij_2x)
            )
        )

        #
        # SHAP uniqueness: upsilon[i, j]
        #

        # 2 * cov(phi[i], epsilon[i, j])
        # shape: (n_targets, n_features, n_features)
        # intermediate result to calculate upsilon[i, j], see next step

        cov_phi_i_epsilon_ij_2x = epsilon_tau_ratio_2x * (
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
        autonomy_ij = std_tau_ij_plus_tau_ji
        equivalence_ij = np.abs(std_epsilon_ij_2x)
        uniqueness_ij = std_upsilon_ij_plus_upsilon_ji
        independence_ij = std_nu_ij_plus_nu_ji

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

    def _reset_fit(self) -> None:
        super()._reset_fit()
        self.synergy_rel_ = self.equivalence_rel_ = None

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


def _ensure_last_axis_is_fast(v: np.ndarray) -> np.ndarray:
    if _PAIRWISE_PARTIAL_SUMMATION:
        if v.strides[-1] != v.itemsize:
            v = v.copy()
        assert v.strides[-1] == v.itemsize
    return v


def _cov(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # calculate covariance matrix of two vectors, assuming µ=0 for both
    # input shape for u and v is (n_targets, n_features, n_observations)
    # output shape is (n_targets, n_features, n_features, n_observations)

    assert u.shape == v.shape
    assert u.ndim == 3

    if _PAIRWISE_PARTIAL_SUMMATION:
        raise NotImplementedError("max precision matmul not yet implemented")
    else:
        return np.matmul(u, v.swapaxes(1, 2)) / u.shape[2]


def _transpose(v: np.ndarray) -> np.ndarray:
    # transpose a feature matrix for all targets
    assert v.ndim == 3

    return v.swapaxes(1, 2)


def _sqrt(v: np.ndarray) -> np.ndarray:
    # we clip values < 0 as these could happen in isolated cases due to
    # rounding errors

    return np.sqrt(np.clip(v, 0, None))
