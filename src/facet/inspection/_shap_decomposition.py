"""
Decomposition of SHAP contribution scores (i.e, SHAP importance) of all possible
pairings of features into additive components for synergy, redundancy, and independence.
"""
import logging
from typing import List, Optional, TypeVar

import numpy as np
import pandas as pd

from pytools.api import AllTracker, deprecated, inheritdoc

from ._shap import ShapCalculator, ShapInteractionValuesCalculator
from ._shap_global_explanation import (
    ShapGlobalExplainer,
    ShapInteractionGlobalExplainer,
    cov,
    ensure_last_axis_is_fast,
    make_symmetric,
    sqrt,
    transpose,
)

log = logging.getLogger(__name__)

__all__ = [
    "ShapDecomposer",
    "ShapInteractionDecomposer",
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
class ShapDecomposer(ShapGlobalExplainer):
    """
    Decomposes SHAP vectors (i.e., SHAP contribution) of all possible pairings
    of features into additive components for association and independence.
    SHAP contribution scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    @deprecated(
        message=(
            "SHAP vector decomposition is deprecated and will be removed in the next "
            "minor release, consider using SHAP vector projection instead"
        )
    )
    def __init__(self) -> None:
        super().__init__()
        self.association_rel_: Optional[np.ndarray] = None
        self.association_rel_asymmetric_: Optional[np.ndarray] = None

    def association(
        self, absolute: bool, symmetrical: bool, std: bool = False
    ) -> Optional[np.ndarray]:
        """[see superclass]"""
        if absolute:
            raise NotImplementedError("absolute association is not supported")
        if std:
            return None

        self._ensure_fitted()
        return (
            self.association_rel_ if symmetrical else self.association_rel_asymmetric_
        )

    # noinspection DuplicatedCode
    def _fit(self, shap_calculator: ShapCalculator) -> None:
        #
        # basic definitions
        #

        shap_values: pd.DataFrame = shap_calculator.get_shap_values(aggregation="mean")
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

        #
        # SHAP association: ass[i, j]
        #

        # covariance matrix of shap vectors
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j = cov(p_i, weight)
        cov_p_i_p_j_2x = 2 * cov_p_i_p_j

        # variance of shap vectors
        var_p = np.diagonal(cov_p_i_p_j, axis1=1, axis2=2)  # (n_outputs, n_features)
        var_p_i = var_p[:, :, np.newaxis]  # (n_outputs, n_features, n_features)
        var_p_i_plus_var_p_j = var_p_i + var_p[:, np.newaxis, :]

        # std(p[i] + p[j])
        # shape: (n_outputs, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors p[i] and p[j]
        # this quantifies the joint contributions of features i and j
        # we also need this as part of the formula to calculate ass_ij (see below)
        std_p_i_plus_p_j = sqrt(var_p_i_plus_var_p_j + cov_p_i_p_j_2x)

        # std(p[i] - p[j])
        # shape: (n_outputs, n_features)
        # the length of the difference of vectors p[i] and p[j]
        # we need this as part of the formula to calculate ass_ij (see below)
        std_p_i_minus_p_j = sqrt(var_p_i_plus_var_p_j - cov_p_i_p_j_2x)

        # 2 * std(ass[i, j]) = 2 * std(ass[j, i])
        # shape: (n_outputs, n_features, n_features)
        # twice the standard deviation (= length) of the redundancy vector;
        # this is the total contribution made by p[i] and p[j] where both features
        # independently use redundant information
        std_ass_ij_2x = std_p_i_plus_p_j - std_p_i_minus_p_j

        #
        # SHAP independence: ind[i, j]
        #

        # 4 * var(ass[i, j]) * var(ass[i, j])
        # shape: (n_outputs, n_features, n_features)
        var_ass_ij_4x = std_ass_ij_2x ** 2

        # ratio of length of 2 * ass over length of (p[i] + p[j])
        # shape: (n_outputs, n_features, n_features)
        # we need this for the next step
        with np.errstate(divide="ignore", invalid="ignore"):
            ass_p_ratio_2x = 1 - std_p_i_minus_p_j / std_p_i_plus_p_j

        # 2 * cov(p[i], ass[i, j])
        # shape: (n_outputs, n_features, n_features)
        # we need this as part of the formula to calculate std(ind_ij)
        # (see next step below)
        cov_p_i_ass_ij_2x = ass_p_ratio_2x * (var_p_i + cov_p_i_p_j)

        # std(ind_ij + ind_ji)
        # where ind_ij + ind_ji = p_i + p_j - 2 * ass_ij
        # shape: (n_outputs, n_features, n_features)
        # the standard deviation (= length) of the independence vector

        std_ind_ij_plus_ind_ji = sqrt(
            var_p_i_plus_var_p_j
            + var_ass_ij_4x
            + 2
            * (
                cov_p_i_p_j
                - cov_p_i_ass_ij_2x
                - transpose(cov_p_i_ass_ij_2x)  # cov_p_j_ass_ij_2x
            )
        )

        # std(ind_ij)
        # where ind_ij = p_i - ass_ij
        # shape: (n_outputs, n_features, n_features)
        # the standard deviation (= length) of the asymmetrical independence vector
        std_ind_ij = sqrt(
            var_p_i + var_ass_ij_4x / 4 - ass_p_ratio_2x * (var_p_i + cov_p_i_p_j)
        )

        # SHAP association
        # shape: (n_outputs, n_features, n_features)
        association_ij = np.abs(std_ass_ij_2x)
        independence_ij = np.abs(std_ind_ij_plus_ind_ji)

        # we should have the right shape for all resulting matrices
        assert association_ij.shape == (n_outputs, n_features, n_features)

        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = association_ij / (association_ij + independence_ij)
            self.association_rel_ = _fill_nans(make_symmetric(matrix))
            self.association_rel_asymmetric_ = _fill_nans(
                std_ass_ij_2x / (std_ass_ij_2x + std_ind_ij * 2)
            )

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.association_rel_ = None


@inheritdoc(match="""[see superclass]""")
class ShapInteractionDecomposer(ShapDecomposer, ShapInteractionGlobalExplainer):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible pairings
    of features into additive components for synergy, redundancy, and independence.
    SHAP interaction scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    #: minimum SHAP direct synergy (as a ratio ranging from 0.0 to 1.0), to be
    #: considered for calculating indirect synergies
    DEFAULT_MIN_DIRECT_SYNERGY = 0.01

    def __init__(self, min_direct_synergy: Optional[float] = None) -> None:
        """
        :param min_direct_synergy: minimum direct synergy a pair of features
            :math:`f_i' and :math:`f_j' needs to manifest in order to be considered
            for calculating indirect synergies. This is expressed as the relative
            contribution score with regard to the total synergistic contributions
            ranging between 0 and 1, and calculated as
            :math:`\\frac \
                    {\\sigma_{\\vec{\\phi_{ij}}}} \
                    {\\sum_{i,j}\\sigma_{\\vec{\\phi_{ij}}}}`,
            i.e, the relative share of the synergy contribution
            :math:`\\sigma_{\\vec{\\phi_{ij}}}`.
        """
        super().__init__()
        self.min_direct_synergy = (
            ShapInteractionDecomposer.DEFAULT_MIN_DIRECT_SYNERGY
            if min_direct_synergy is None
            else min_direct_synergy
        )
        self.synergy_rel_: Optional[np.ndarray] = None
        self.redundancy_rel_: Optional[np.ndarray] = None
        self.synergy_rel_asymmetric_: Optional[np.ndarray] = None
        self.redundancy_rel_asymmetric_: Optional[np.ndarray] = None

    __init__.__doc__ += f"""\
            (default: {DEFAULT_MIN_DIRECT_SYNERGY}, i.e.,
            {DEFAULT_MIN_DIRECT_SYNERGY * 100.0:g}%)
        """

    def synergy(
        self, symmetrical: bool, absolute: bool, std: bool = False
    ) -> Optional[np.ndarray]:
        """[see superclass]"""
        if absolute:
            raise NotImplementedError("absolute synergy is not supported")
        if std:
            return None

        self._ensure_fitted()
        return self.synergy_rel_ if symmetrical else self.synergy_rel_asymmetric_

    def redundancy(
        self, symmetrical: bool, absolute: bool, std: bool = False
    ) -> Optional[np.ndarray]:
        """[see superclass]"""
        if absolute:
            raise NotImplementedError("absolute redundancy is not supported")
        if std:
            return None

        self._ensure_fitted()
        return self.redundancy_rel_ if symmetrical else self.redundancy_rel_asymmetric_

    # noinspection DuplicatedCode
    def _fit(self, shap_calculator: ShapInteractionValuesCalculator) -> None:
        super()._fit(shap_calculator)

        #
        # basic definitions
        #
        shap_values: pd.DataFrame = shap_calculator.get_shap_interaction_values(
            aggregation="mean"
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

        # p[i]
        # shape: (n_outputs, n_features, n_observations)
        p_i = ensure_last_axis_is_fast(p_ij.sum(axis=2))

        # cov(p[i], p[j])
        # covariance matrix of shap vectors
        # shape: (n_outputs, n_features, n_features)
        cov_p_i_p_j = cov(p_i, weight)

        #
        # Feature synergy (direct and indirect): zeta[i, j]
        #

        # var(p[i, j]), std(p[i, j])
        # shape: (n_outputs, n_features, n_features)
        # variance and length (= standard deviation) of each feature interaction vector
        var_p_ij = np.average(
            ensure_last_axis_is_fast(p_ij ** 2), axis=-1, weights=weight
        )
        std_p_ij = np.sqrt(var_p_ij)

        # p[i, i]
        # shape: (n_outputs, n_features, n_observations)
        # independent feature contributions;
        # this is the diagonal of p_ij
        p_ii = np.diagonal(p_ij, axis1=1, axis2=2).swapaxes(1, 2)

        # p'[i] = p[i] - p[i, i]
        # shape: (n_outputs, n_features, n_observations)
        # the SHAP vectors per feature, minus the independent contributions
        p_prime_i = ensure_last_axis_is_fast(p_i - p_ii)

        # std_p_relative[i, j]
        # shape: (n_outputs, n_features, n_features)
        # relative importance of p[i, j] measured as the length of p[i, j]
        # as percentage of the sum of lengths of all p[..., ...]
        with np.errstate(divide="ignore", invalid="ignore"):
            std_p_relative_ij = std_p_ij / std_p_ij.sum()

        # p_valid[i, j]
        # shape: (n_outputs, n_features, n_features)
        # boolean values indicating whether p[i, j] is above the "noise" threshold,
        # i.e. whether we trust that doing calculations with p[i, j] is sufficiently
        # accurate. p[i, j] with small variances should not be used because we divide
        # by that variance to determine the multiple for indirect synergy, which can
        # deviate significantly if var(p[i, j]) is only slightly off

        interaction_noise_threshold = self.min_direct_synergy
        p_valid_ij = std_p_relative_ij >= interaction_noise_threshold

        # k[i, j]
        # shape: (n_outputs, n_features, n_features)
        # k[i, j] = cov(p'[i], p[i, j]) / var(p[i, j]), for each output
        # this is the orthogonal projection of p[i, j] onto p'[i] and determines
        # the multiplier of std(p[i, j]) (i.e., direct synergy) to obtain
        # the total direct and indirect synergy of p'[i] with p'[j], i.e.,
        # * k[i, j] * std(p[i, j])

        if _PAIRWISE_PARTIAL_SUMMATION:
            raise NotImplementedError(
                "max precision Einstein summation not yet implemented"
            )
        # noinspection SpellCheckingInspection,PyTypeChecker
        k_ij = np.divide(
            # cov(p'[i], p[i, j])
            np.einsum(
                "tio,tijo->tij",
                (
                    p_prime_i
                    if weight is None
                    else p_prime_i * weight.reshape((1, 1, -1))
                ),
                p_ij,
                optimize=True,
            )
            / (n_observations if weight is None else weight.sum()),
            # var(p[i, j])
            var_p_ij,
            out=np.ones_like(var_p_ij),
            where=p_valid_ij,
        )

        # issue warning messages for edge cases

        def _feature(_i: int) -> str:
            return f'"{features[_i]}"'

        def _for_output(_t: int) -> str:
            if outputs is None:
                return ""
            else:
                return f' for output "{outputs[_t]}"'

        def _relative_direct_synergy(_t: int, _i: int, _j: int) -> str:
            return (
                f"p[{features[_i]}, {features[_j]}] has "
                f"{std_p_relative_ij[_t, _i, _j] * 100:.3g}% "
                "relative SHAP contribution; "
                "consider increasing the interaction noise threshold (currently "
                f"{interaction_noise_threshold * 100:.3g}%). "
            )

        def _test_synergy_feasibility() -> None:
            for _t, _i, _j in np.argwhere(k_ij < 1):
                if _i != _j:
                    log.debug(
                        "contravariant indirect synergy "
                        f"between {_feature(_i)} and {_feature(_j)}{_for_output(_t)}: "
                        "indirect synergy calculated as "
                        f"{(k_ij[_t, _i, _j] - 1) * 100:.3g}% "
                        "of direct synergy; setting indirect synergy to 0. "
                        f"{_relative_direct_synergy(_t, _i, _j)}"
                    )

            for _t, _i, _j in np.argwhere(k_ij - 1 > np.log2(n_features)):
                if _i != _j:
                    log.warning(
                        "high indirect synergy "
                        f"between {_feature(_i)} and {_feature(_j)}{_for_output(_t)}: "
                        "total synergy is "
                        f"{k_ij[_t, _i, _j] * 100:.3g}% of direct synergy. "
                        f"{_relative_direct_synergy(_t, _i, _j)}"
                    )

        _test_synergy_feasibility()

        # ensure that k[i, j] is at least 1.0
        # (a warning will have been issued during the checks above for k[i, j] < 1)
        # i.e. we don't allow total synergy to be less than direct synergy
        k_ij = np.clip(k_ij, 1.0, None)

        # fill the diagonal(s) with nan since these values are meaningless
        for k_ij_for_output in k_ij:
            np.fill_diagonal(k_ij_for_output, val=np.nan)

        # syn[i, j] = k[i, j] * p[i, j]
        std_syn_ij = k_ij * std_p_ij

        # k[j, i]
        # transpose of k[i, j]; we need this later for calculating SHAP redundancy
        # shape: (n_outputs, n_features, n_features)
        k_ji = transpose(k_ij)

        # syn[i, j] + syn[j, i] = (k[i, j] + k[j, i]) * p[i, j]
        # total SHAP synergy, comprising both direct and indirect synergy
        # shape: (n_outputs, n_features, n_features)
        std_syn_ij_plus_syn_ji = (k_ij + k_ji) * std_p_ij

        #
        # SHAP autonomy: aut[i, j]
        #

        # cov(p[i], p[i, j])
        # covariance matrix of shap vectors with pairwise synergies
        # shape: (n_outputs, n_features, n_features)

        if _PAIRWISE_PARTIAL_SUMMATION:
            raise NotImplementedError(
                "max precision Einstein summation not yet implemented"
            )
        cov_p_i_p_ij = np.einsum("...io,...ijo->...ij", p_i, p_ij) / n_observations

        # cov(aut[i, j], aut[j, i])
        # where aut[i, j] = p[i] - k[i, j] * p[i, j]
        # shape: (n_observations, n_outputs, n_features)
        # matrix of covariances for autonomy vectors for all pairings of features
        # the aut[i, j] vector is the p[i] SHAP contribution vector where the synergy
        # effects (direct and indirect) with feature j have been deducted

        cov_aut_ij_aut_ji = (
            cov_p_i_p_j
            - k_ji * cov_p_i_p_ij
            - k_ij * transpose(cov_p_i_p_ij)
            + k_ji * k_ij * var_p_ij
        )

        # var(p[i])
        # variances of SHAP vectors
        # shape: (n_outputs, n_features, 1)
        # i.e. adding a second, empty feature dimension to enable correct broadcasting
        var_p_i = np.diagonal(cov_p_i_p_j, axis1=1, axis2=2)[:, :, np.newaxis]

        # var(aut[i, j])
        # variances of SHAP vectors minus total synergy
        # shape: (n_outputs, n_features, n_features)
        var_aut_ij = var_p_i - 2 * k_ij * cov_p_i_p_ij + k_ij * k_ij * var_p_ij

        # std(aut[i, j])
        std_aut_ij = sqrt(var_aut_ij)

        # var(aut[i]) + var(aut[j])
        # Sum of covariances per feature pair (this is a diagonal matrix)
        # shape: (n_outputs, n_features, n_features)
        var_aut_ij_plus_var_aut_ji = var_aut_ij + transpose(var_aut_ij)

        # 2 * cov(aut[i, j], aut[j, i])
        # shape: (n_outputs, n_features, n_features)
        # this is an intermediate result to calculate the standard deviation
        # of the redundancy vector (see next step below)
        cov_aut_ij_aut_ji_2x = 2 * cov_aut_ij_aut_ji

        # std(aut[i, j] + aut[j, i])
        # where aut[i, j] = p[i] - syn[i, j]
        # shape: (n_outputs, n_features, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors aut[i, j] and aut[j, i]
        # this quantifies the autonomous contributions of features i and j, i.e.,
        # without synergizing
        # we also need this as part of the formula to calculate red_ij (see below)

        std_aut_ij_plus_aut_ji = sqrt(var_aut_ij_plus_var_aut_ji + cov_aut_ij_aut_ji_2x)

        #
        # SHAP redundancy: red[i, j]
        #

        # std(aut[i, j] - aut[j, i])
        # shape: (n_outputs, n_features, n_features)
        # the length of the difference of vectors aut[i, j] and aut[j, i]
        # we need this as part of the formula to calculate red_ij (see below)

        std_aut_ij_minus_aut_ji = sqrt(
            var_aut_ij_plus_var_aut_ji - cov_aut_ij_aut_ji_2x
        )

        # 2 * std(red[i, j]) = 2 * std(red[j, i])
        #   = std(aut[i, j] + aut[j, i]) - std(aut[i, j] - aut[j, i])
        # shape: (n_outputs, n_features)
        # twice the standard deviation (= length) of redundancy vector;
        # this is the total contribution made by p[i] and p[j] where both features
        # independently use redundant information

        std_red_ij_2x = np.abs(std_aut_ij_plus_aut_ji - std_aut_ij_minus_aut_ji)

        #
        # SHAP independence: ind[i, j]
        #

        # 4 * var(red[i, j]), var(red[i, j])
        # shape: (n_outputs, n_features, n_features)
        var_red_ij_4x = std_red_ij_2x ** 2

        # ratio of length of 2 * red over length of (aut[i, j] + aut[j, i])
        # shape: (n_outputs, n_features, n_features)
        # we need this for the next step
        with np.errstate(divide="ignore", invalid="ignore"):
            red_aut_ratio_2x = 1 - std_aut_ij_minus_aut_ji / std_aut_ij_plus_aut_ji

        # 2 * cov(aut[i, j], red[i, j])
        # shape: (n_outputs, n_features, n_features)
        # we need this as part of the formula to calculate std(ind_ij)
        # (see next step below)
        cov_aut_ij_red_ij_2x = red_aut_ratio_2x * (var_aut_ij + cov_aut_ij_aut_ji)

        # std(ind_ij + ind_ji)
        # where ind_ij + ind_ji = aut_ij + aut_ji - 2 * red_ij
        # shape: (n_outputs, n_features, n_features)
        # the standard deviation (= length) of the combined independence vectors

        std_ind_ij_plus_ind_ji = sqrt(
            var_aut_ij
            + transpose(var_aut_ij)
            + var_red_ij_4x
            + 2
            * (
                cov_aut_ij_aut_ji
                - cov_aut_ij_red_ij_2x
                - transpose(cov_aut_ij_red_ij_2x)
            )
        )

        #
        # SHAP uniqueness: uni[i, j]
        #

        # 2 * cov(p[i], red[i, j])
        # shape: (n_outputs, n_features, n_features)
        # intermediate result to calculate uni[i, j], see next step

        cov_p_i_red_ij_2x = red_aut_ratio_2x * (
            var_p_i + cov_p_i_p_j - (k_ij + k_ji) * cov_p_i_p_ij
        )

        # std(uni[i, j]) = std(p[i] - red[i, j])
        # where uni[i, j] = p[i] - red[i, j]
        # shape: (n_outputs, n_features, n_features)
        # this is the sum of complementary contributions of feature i w.r.t. feature j,
        # i.e., deducting the redundant contributions
        std_uni_ij = sqrt(var_p_i + var_red_ij_4x / 4 - cov_p_i_red_ij_2x)

        # std(uni[i, j] + uni[j, i])
        # where uni[i, j] + uni[j, i] = p[i] + p[j] - 2 * red[i, j]
        # shape: (n_outputs, n_features, n_features)
        # this is the sum of complementary contributions of feature i and feature j,
        # i.e., deducting the redundant contributions

        std_uni_ij_plus_uni_ji = sqrt(
            var_p_i
            + transpose(var_p_i)
            + var_red_ij_4x
            + 2 * (cov_p_i_p_j - cov_p_i_red_ij_2x - transpose(cov_p_i_red_ij_2x))
        )

        #
        # SHAP decomposition as relative contributions of
        # synergy, redundancy, and independence
        #

        synergy_ij = std_syn_ij_plus_syn_ji
        autonomy_ij = std_aut_ij_plus_aut_ji
        redundancy_ij = std_red_ij_2x
        uniqueness_ij = std_uni_ij_plus_uni_ji
        independence_ij = std_ind_ij_plus_ind_ji

        # we should have the right shape for all resulting matrices
        for matrix in (
            synergy_ij,
            redundancy_ij,
            autonomy_ij,
            uniqueness_ij,
            independence_ij,
        ):
            assert matrix.shape == (n_outputs, n_features, n_features)

        # Calculate relative synergy and redundancy (ranging from 0.0 to 1.0),
        # as a symmetric and an asymmetric measure.
        # For the symmetric case, we ensure perfect symmetry by removing potential
        # round-off errors
        # NOTE: we do not store independence, so technically it could be removed from
        # the code above

        with np.errstate(divide="ignore", invalid="ignore"):
            matrix1 = synergy_ij / (synergy_ij + autonomy_ij)
            self.synergy_rel_ = _fill_nans(make_symmetric(matrix1))
            matrix2 = redundancy_ij / (redundancy_ij + uniqueness_ij)
            self.redundancy_rel_ = _fill_nans(make_symmetric(matrix2))
            self.synergy_rel_asymmetric_ = _fill_nans(
                std_syn_ij / (std_syn_ij + std_aut_ij)
            )
            self.redundancy_rel_asymmetric_ = _fill_nans(
                std_red_ij_2x / (std_red_ij_2x + std_uni_ij * 2)
            )

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.synergy_rel_ = None
        self.redundancy_rel_ = None
        self.synergy_rel_asymmetric_ = None
        self.redundancy_rel_asymmetric_ = None


def _fill_nans(matrix: np.ndarray) -> np.ndarray:
    # apply fixes in-place for each output
    for m in matrix:
        # set the matrix diagonals to 1.0 = full association of each feature with
        # itself
        np.fill_diagonal(m, 1.0)

        # replace nan values with 0.0 = no association when correlation is undefined
        np.nan_to_num(m, copy=False)

    return matrix


__tracker.validate()
