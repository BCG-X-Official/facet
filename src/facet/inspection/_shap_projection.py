"""
Projection of SHAP contribution scores (i.e, SHAP importance) of all possible
pairings of features onto the SHAP importance vector in partitions of for synergy,
redundancy, and independence.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin, fitted_only

from ._shap_context import (
    AffinityMatrix,
    ShapContext,
    ShapInteractionValueContext,
    ShapValueContext,
    cov,
    cov_broadcast,
    ensure_last_axis_is_fast,
    fill_diagonal,
    sqrt,
    transpose,
)
from .shap import ShapCalculator

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
T_Projector = TypeVar("T_Projector", bound="ShapProjector")

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class ShapProjector(FittableMixin[ShapCalculator[Any]], metaclass=ABCMeta):
    """
    Base class for global pairwise model explanations based on SHAP vector projection.

    Derives feature association as a global metric of SHAP values for multiple
    observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_index_: Optional[pd.Index] = None
        self.association_: Optional[AffinityMatrix] = None

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.feature_index_ is not None

    def fit(  # type: ignore[override]
        self: T_Projector,
        shap_calculator: ShapCalculator[Any],
        *,
        sample_weight: Optional[pd.Series] = None,
        **fit_params: Any,
    ) -> T_Projector:
        """
        Calculate the SHAP decomposition for the shap values produced by the
        given SHAP calculator.

        :param shap_calculator: the fitted calculator from which to get the shap values
        :param sample_weight: optional sample weights to apply for the global
            explanations; the index must match the index of the features used to
            fit the SHAP calculator
        """

        self._reset_fit()

        if len(fit_params) > 0:
            raise ValueError(
                f'unsupported fit parameters: {", ".join(fit_params.values())}'
            )

        self._calculate(
            self._get_context(
                shap_calculator=shap_calculator, sample_weight=sample_weight
            )
        )

        self.feature_index_ = shap_calculator.feature_index_

        return self

    @fitted_only
    def association(self, absolute: bool, symmetrical: bool) -> npt.NDArray[np.float_]:
        """
        The association matrix for all feature pairs.

        Raises an error if this global explainer has not been fitted.

        :param absolute: if ``False``, return relative association as a percentage of
            total feature importance;
            if ``True``, return absolute association as a portion of feature importance
        :param symmetrical: if ``False``, return an asymmetrical matrix
            quantifying unilateral association of the features represented by rows
            with the features represented by columns;
            if ``True``, return a symmetrical matrix quantifying mutual association
        :returns: the matrix as an array of shape (n_outputs, n_features, n_features)
        """

        assert self.association_ is not None
        return self.association_.get_values(symmetrical=symmetrical, absolute=absolute)

    def to_frames(self, matrix: npt.NDArray[np.float_]) -> List[pd.DataFrame]:
        """
        Transforms one or more affinity matrices into a list of data frames.

        :param matrix: an array of shape `(n_outputs, n_features, n_features)`,
            representing one or more affinity matrices
        :return: a list of `n_outputs` data frames of shape `(n_features, n_features)`
        """
        assert self.feature_index_ is not None, "explainer is fitted"
        index = self.feature_index_

        n_features = len(index)
        assert matrix.ndim == 3
        assert matrix.shape[1:] == (n_features, n_features)

        return [
            pd.DataFrame(
                m,
                index=index,
                columns=index,
            )
            for m in matrix
        ]

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        self.feature_index_ = None
        self.association_ = None

    @abstractmethod
    def _get_context(
        self, shap_calculator: ShapCalculator[Any], sample_weight: Optional[pd.Series]
    ) -> ShapContext:
        pass

    @abstractmethod
    def _calculate(self, context: ShapContext) -> None:
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

        # association of a feature with itself is undefined
        fill_diagonal(ass_ij, np.nan)

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

    def _get_context(
        self, shap_calculator: ShapCalculator[Any], sample_weight: Optional[pd.Series]
    ) -> ShapContext:
        return ShapValueContext(
            shap_calculator=shap_calculator, sample_weight=sample_weight
        )

    def _calculate(self, context: ShapContext) -> None:
        # calculate association matrices for each SHAP context, then aggregate
        self.association_ = self._calculate_association(context)


class ShapInteractionVectorProjector(ShapProjector):
    """
    Derives feature association, synergy, and redundancy as a global metric of SHAP
    interaction values for multiple observations.

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

    @fitted_only
    def synergy(self, symmetrical: bool, absolute: bool) -> npt.NDArray[np.float_]:
        """
        The synergy matrix for all feature pairs.

        Raises an error if this global explainer has not been fitted.

        :param absolute: if ``False``, return relative synergy as a percentage of
            total feature importance;
            if ``True``, return absolute synergy as a portion of feature importance
        :param symmetrical: if ``False``, return an asymmetrical matrix
            quantifying unilateral synergy of the features represented by rows
            with the features represented by columns;
            if ``True``, return a symmetrical matrix quantifying mutual synergy
        :returns: the matrix as an array of shape (n_outputs, n_features, n_features)
        """

        assert self.synergy_ is not None, "Projector is fitted"
        return self.synergy_.get_values(symmetrical=symmetrical, absolute=absolute)

    @fitted_only
    def redundancy(self, symmetrical: bool, absolute: bool) -> npt.NDArray[np.float_]:
        """
        The redundancy matrix for all feature pairs.

        Raises an error if this global explainer has not been fitted.

        :param absolute: if ``False``, return relative redundancy as a percentage of
            total feature importance;
            if ``True``, return absolute redundancy as a portion of feature importance
        :param symmetrical: if ``False``, return an asymmetrical matrix
            quantifying unilateral redundancy of the features represented by rows
            with the features represented by columns;
            if ``True``, return a symmetrical matrix quantifying mutual redundancy
        :returns: the matrix as an array of shape (n_outputs, n_features, n_features)
        """

        assert self.redundancy_ is not None, "Projector is fitted"
        return self.redundancy_.get_values(symmetrical=symmetrical, absolute=absolute)

    def _get_context(
        self, shap_calculator: ShapCalculator[Any], sample_weight: Optional[pd.Series]
    ) -> ShapContext:
        return ShapInteractionValueContext(
            shap_calculator=shap_calculator, sample_weight=sample_weight
        )

    def _calculate(self, context: ShapContext) -> None:
        # calculate association, synergy, and redundancy matrices for the SHAP context

        self.association_ = self._calculate_association(context=context)
        self.synergy_, self.redundancy_ = self._calculate_synergy_redundancy(
            context=context
        )

    @staticmethod
    def _calculate_synergy_redundancy(
        context: ShapContext,
    ) -> Tuple[AffinityMatrix, AffinityMatrix]:
        p_i = context.p_i
        var_p_i = context.var_p_i
        assert context.p_ij is not None, "Projector has interaction values enabled"
        p_ij = context.p_ij
        weight = context.weight

        #
        # Synergy: syn[i, j]
        #

        # var(p[i, j])
        # shape: (n_outputs, n_features, n_features)
        # variance of each feature interaction vector
        var_p_ij = np.average(
            ensure_last_axis_is_fast(p_ij**2), axis=-1, weights=weight
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

        # synergy of a feature with itself is undefined
        fill_diagonal(syn_ij, np.nan)

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

        # redundancy of a feature with itself is undefined
        fill_diagonal(red_ij, np.nan)

        #
        # SHAP decomposition as relative contributions of
        # synergy, redundancy, and independence
        #

        # we should have the right shape for all resulting matrices
        assert syn_ij.shape == matrix_shape
        assert red_ij.shape == matrix_shape

        # Calculate relative synergy and redundancy (ranging from 0.0 to 1.0),
        # as a symmetric and an asymmetric measure.
        #
        # For the symmetric case, we ensure perfect symmetry by removing potential
        # round-off errors.
        #
        # NOTE: we do not store independence, so technically it could be removed from
        # the code above.

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
