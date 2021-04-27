"""
Projection of SHAP contribution scores (i.e, SHAP importance) of all possible
pairings of features onto the SHAP importance vector in partitions of for synergy,
redundancy, and independence.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin

from ._shap import ShapCalculator

log = logging.getLogger(__name__)

__all__ = [
    "AffinityMatrix",
    "ShapContext",
    "ShapGlobalExplainer",
    "ShapInteractionGlobalExplainer",
    "ShapInteractionValueContext",
    "ShapValueContext",
    "cov",
    "cov_broadcast",
    "diagonal",
    "ensure_last_axis_is_fast",
    "fill_diagonal",
    "make_symmetric",
    "sqrt",
    "transpose",
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
T_ShapCalculator = TypeVar("T_ShapCalculator", bound=ShapCalculator)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class AffinityMatrix:
    """
    Stores all variations of a feature affinity matrix.
    """

    # shape: (2, 2, n_outputs, n_features, n_features)
    _matrices: np.ndarray

    # shape: (2, 2, n_outputs, n_features, n_features)
    _matrices_std: Optional[np.ndarray]

    def __init__(
        self, matrices: np.ndarray, matrices_std: Optional[np.ndarray] = None
    ) -> None:
        shape = matrices.shape
        assert len(shape) == 5
        assert shape[:2] == (2, 2)
        assert shape[3] == shape[4]
        assert matrices_std is None or matrices_std.shape == matrices.shape

        self._matrices = matrices
        self._matrices_std = matrices_std

    @staticmethod
    def from_relative_affinity(
        affinity_rel_ij: np.ndarray, std_p_i: np.ndarray
    ) -> "AffinityMatrix":
        """
        :param affinity_rel_ij: the affinity matrix from which to create all variations,
            shaped `(n_outputs, n_features, n_features)`
        :param std_p_i: SHAP vector magnitudes for all outputs and features,
            shaped `(n_outputs, n_features, 1)`
        """
        assert affinity_rel_ij.ndim == 3
        assert std_p_i.ndim == 3
        assert affinity_rel_ij.shape[:2] == std_p_i.shape[:2]
        assert affinity_rel_ij.shape[1] == affinity_rel_ij.shape[2]
        assert std_p_i.shape[2] == 1

        # normalize SHAP vector magnitudes to get feature importance in %
        importance_ij = std_p_i / std_p_i.sum(axis=1).reshape(std_p_i.shape[0], 1, 1)

        # absolute affinity is relative affinity scaled by feature importance (row-wise)
        affinity_abs_ij = importance_ij * affinity_rel_ij

        # absolute symmetrical affinity is the mean of unilateral absolute affinity
        affinity_abs_sym_ij_2x = affinity_abs_ij + transpose(affinity_abs_ij)

        # relative symmetrical affinity is absolute symmetrical affinity scaled back
        # from total feature importance per feature pair
        affinity_rel_sym_ij = np.zeros(affinity_rel_ij.shape)
        np.divide(
            affinity_abs_sym_ij_2x,
            importance_ij + transpose(importance_ij),
            out=affinity_rel_sym_ij,
            # do not divide where the nominator is 0 (the denominator will be 0 as well)
            where=affinity_abs_sym_ij_2x > 0.0,
        )

        # re-set the diagonal to 1.0 in case of rounding errors
        fill_diagonal(affinity_rel_sym_ij, 1.0)

        # return the AffinityMatrices object
        return AffinityMatrix(
            matrices=np.vstack(
                (
                    affinity_rel_ij,
                    affinity_abs_ij,
                    affinity_rel_sym_ij,
                    affinity_abs_sym_ij_2x / 2,
                )
            ).reshape((2, 2, *affinity_rel_ij.shape))
        )

    @staticmethod
    def aggregate(affinity_matrices: Iterable["AffinityMatrix"]) -> "AffinityMatrix":
        """
        Aggregate several sets of affinity matrices (obtained from different splits)
        into one, by calculating the mean and standard deviation for each value in the
        provided iterable of affinity matrices.

        :param affinity_matrices: sets of affinity matrices to aggregate
        :return: the aggregated set of affinity matrices
        """
        matrix_values = np.stack(
            tuple(affinity_matrix._matrices for affinity_matrix in affinity_matrices)
        )
        return AffinityMatrix(
            matrices=matrix_values.mean(axis=0), matrices_std=matrix_values.std(axis=0)
        )

    def get_values(
        self, symmetrical: bool, absolute: bool, std: bool
    ) -> Optional[np.ndarray]:
        """
        Get the matrix matching the given criteria.
        :param symmetrical: if ``True``, get the symmetrical version of the matrix
        :param absolute: if ``True``, get the absolute version of the matrix
        :param std: if ``True``, return standard deviations instead of (mean) values;
            return ``None`` if only a single affinity matrix had been calculated and
            thus the standard deviation is not known
        :return: the affinity matrix
        """
        if std:
            matrices = self._matrices_std
            if matrices is None:
                return None
        else:
            matrices = self._matrices
        return matrices[int(symmetrical), int(absolute)]


@inheritdoc(match="""[see superclass]""")
class ShapGlobalExplainer(FittableMixin[ShapCalculator], metaclass=ABCMeta):
    """
    Derives feature association as a global metric of SHAP values for multiple
    observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_index_: Optional[pd.Index] = None

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.feature_index_ is not None

    def fit(self: T_Self, shap_calculator: ShapCalculator, **fit_params: Any) -> T_Self:
        """
        Calculate the SHAP decomposition for the shap values produced by the
        given SHAP calculator.

        :param shap_calculator: the fitted calculator from which to get the shap values
        """

        self: ShapGlobalExplainer  # support type hinting in PyCharm

        try:
            if len(fit_params) > 0:
                raise ValueError(
                    f'unsupported fit parameters: {", ".join(fit_params.values())}'
                )

            self._fit(shap_calculator=shap_calculator)

            self.feature_index_ = shap_calculator.feature_index_

        except Exception:
            # reset fit in case we get an exception along the way
            self._reset_fit()
            raise

        return self

    @abstractmethod
    def association(
        self, absolute: bool, symmetrical: bool, std: bool = False
    ) -> Optional[np.ndarray]:
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
        :param std: if ``True``, return a matrix of estimated standard deviations
            instead of (mean) values; return ``None`` if the matrix was determined
            from a single model and thus no standard deviation could be estimated
        :returns: the matrix as an array of shape (n_outputs, n_features, n_features)
        """

    def to_frames(self, matrix: np.ndarray) -> List[pd.DataFrame]:
        """
        Transforms one or more affinity matrices into a list of data frames.

        :param matrix: an array of shape `(n_outputs, n_features, n_features)`,
            representing one or more affinity matrices
        :return: a list of `n_outputs` data frames of shape `(n_features, n_features)`
        """
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

    @abstractmethod
    def _fit(self, shap_calculator: ShapCalculator) -> None:
        pass

    def _reset_fit(self) -> None:
        self.feature_index_ = None


class ShapInteractionGlobalExplainer(ShapGlobalExplainer, metaclass=ABCMeta):
    """
    Derives feature association, synergy, and redundancy as a global metric of SHAP
    interaction values for multiple observations.
    """

    @abstractmethod
    def synergy(
        self, symmetrical: bool, absolute: bool, std: bool = False
    ) -> Optional[np.ndarray]:
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
        :param std: if ``True``, return a matrix of estimated standard deviations
            instead of (mean) values; return ``None`` if the matrix was determined
            from a single model and thus no standard deviation could be estimated
        :returns: the matrix as an array of shape (n_outputs, n_features, n_features)
        """

    @abstractmethod
    def redundancy(
        self, symmetrical: bool, absolute: bool, std: bool = False
    ) -> Optional[np.ndarray]:
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
        :param std: if ``True``, return a matrix of estimated standard deviations
            instead of (mean) values; return ``None`` if the matrix was determined
            from a single model and thus no standard deviation could be estimated
        :returns: the matrix as an array of shape (n_outputs, n_features, n_features)
        """


#
# Utility functions
#


def ensure_last_axis_is_fast(array: np.ndarray) -> np.ndarray:
    """
    For future implementations, ensure that the last axis of the given array is `fast`
    to allow for `partial summation`.
    This will be relevant once ``np.matmul`` and ``np.einsum`` support partial
    summation.

    :param array: a numpy array
    :return: an equivalent array where the last axis is guaranteed to be `fast`
    """
    if _PAIRWISE_PARTIAL_SUMMATION:
        if array.strides[-1] != array.itemsize:
            array = array.copy()
        assert array.strides[-1] == array.itemsize
    return array


def sqrt(array: np.ndarray) -> np.ndarray:
    """
    Get the square root of each element in the given array.

    Negative values are replaced by `0` before calculating the square root, to prevent
    errors from minimally negative values due to rounding errors.

    :param array: an arbitrary array
    :return: array of same shape as arg ``array``, with all values replaced by their
        square root
    """

    return np.sqrt(np.clip(array, 0, None))


def make_symmetric(m: np.ndarray) -> np.ndarray:
    """
    Enforce matrix symmetry by transposing the `feature x feature` matrix for each
    output and averaging it with the original matrix.

    :param m: array of shape `(n_outputs, n_features, n_features)`
    :return: array of shape `(n_outputs, n_features, n_features)` with `n_outputs`
        symmetrical `feature x feature` matrices
    """
    return (m + transpose(m)) / 2


def transpose(m: np.ndarray, ndim: int = 3) -> np.ndarray:
    """
    Transpose the `feature x feature` matrix for each output.

    Supports matrices with identical values per row, represented as a broadcastable
    `numpy` array of shape `(n_features, 1)`.

    :param m: array of shape `(n_outputs, n_features, n_features)`
        or shape `(n_outputs, n_features, n_features, n_observations)`
        or shape `(n_outputs, n_features, 1)`
        or shape `(n_outputs, n_features, 1, n_observations)`
    :param ndim: expected dimensions of ``m`` for validation purposes
    :return: array of same shape as arg ``m``, with both feature axes swapped
    """
    assert m.ndim == ndim
    assert m.shape[1] == m.shape[2] or m.shape[2] == 1

    return m.swapaxes(1, 2)


def diagonal(m: np.ndarray) -> np.ndarray:
    """
    Get the diagonal of the `feature x feature` matrix for each output.

    :param m: array of shape `(n_outputs, n_features, n_features)`
    :return: array of shape `(n_outputs, n_features)`, with the diagonals of arg ``m``
    """
    assert m.ndim == 3
    assert m.shape[1] == m.shape[2]
    return m.diagonal(axis1=1, axis2=2)


def fill_diagonal(m: np.ndarray, value: Union[float, np.ndarray]) -> None:
    """
    Fill the diagonal of the `feature x feature` matrix for each output with the given
    value.

    :param m: array of shape `(n_outputs, n_features, n_features)`
    :param value: scalar or array of shape `(n_features)` to fill each diagonal with
    """
    assert m.ndim == 3
    assert m.shape[1] == m.shape[2]
    if isinstance(value, np.ndarray):
        assert value.ndim == 2 and value.shape[:2] == m.shape[:2]
        for m_i, value_i in zip(m, value):
            np.fill_diagonal(m_i, value_i, wrap=True)
    else:
        for m_i in m:
            np.fill_diagonal(m_i, value, wrap=True)


def cov(vectors: np.ndarray, weight: Optional[np.ndarray]) -> np.ndarray:
    """
    Calculate the covariance matrix of pairs of vectors along the observations axis and
    for each output, assuming all vectors are centered (µ=0).

    :param vectors: a sequence of `n_features` vectors per output,
        shaped `(n_outputs, n_features, n_observations)`
    :param weight: an optional array with weights per observation
        shaped `(n_observations)`
    :return: covariance matrices for each output,
        shaped `(n_outputs, n_features, n_features)`
    """
    assert vectors.ndim == 3
    assert weight is None or vectors.shape[2:] == weight.shape

    if _PAIRWISE_PARTIAL_SUMMATION:
        raise NotImplementedError("max precision matmul not yet implemented")

    if weight is None:
        vectors_weighted = vectors
        weight_total = vectors.shape[2]
    else:
        vectors_weighted = vectors * weight.reshape((1, 1, -1))
        weight_total = weight.sum()

    return np.matmul(vectors_weighted, vectors.swapaxes(1, 2)) / weight_total


def cov_broadcast(
    vector_sequence: np.ndarray, vector_grid: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    """
    Calculate the covariance matrix between a sequence of vectors and a grid of vectors
    along the observations axis and for each output, assuming all vectors are centered
    (µ=0).

    :param vector_sequence: a sequence of `n_features` vectors per output,
        shaped `(n_outputs, n_features, n_observations)`
    :param vector_grid: a grid of `n_features x n_features` vectors per output,
        shaped `(n_outputs, n_features, n_features, n_observations)`
    :param weight: an optional array with weights per observation
        shaped `(n_observations)`
    :return: covariance matrices for each output,
        shaped `(n_outputs, n_features, n_features)`
    """
    assert vector_sequence.ndim == 3
    assert vector_grid.ndim == 4
    assert (
        tuple(vector_sequence.shape[i] for i in (0, 1, 1, 2)) == vector_grid.shape
    ), f"shapes {vector_sequence.shape} and {vector_grid.shape} are compatible"
    assert weight is None or vector_sequence.shape[2:] == weight.shape

    if _PAIRWISE_PARTIAL_SUMMATION:
        raise NotImplementedError(
            "max precision Einstein summation not yet implemented"
        )

    if weight is None:
        vectors_weighted = vector_sequence
        weight_total = vector_sequence.shape[2]
    else:
        vectors_weighted = vector_sequence * weight.reshape((1, 1, -1))
        weight_total = weight.sum()

    return (
        np.einsum("...io,...ijo->...ij", vectors_weighted, vector_grid) / weight_total
    )


class ShapContext(metaclass=ABCMeta):
    """
    Contextual data for global SHAP calculations.
    """

    #: SHAP vectors,
    #: with shape `(n_outputs, n_features, n_observations)`
    p_i: np.ndarray

    #: observation weights (optional),
    #: with shape `(n_observations)`
    weight: Optional[np.ndarray]

    #: Covariance matrix for p[i],
    #: with shape `(n_outputs, n_features, n_features)`
    cov_p_i_p_j: np.ndarray

    #: Variances for p[i],
    #: with shape `(n_outputs, n_features, 1)`
    var_p_i: np.ndarray

    #: SHAP interaction vectors,
    #: with shape `(n_outputs, n_features, n_features, n_observations)`
    p_ij: Optional[np.ndarray]

    def __init__(
        self,
        p_i: np.ndarray,
        p_ij: Optional[np.ndarray],
        weight: Optional[np.ndarray],
    ) -> None:
        assert p_i.ndim == 3
        if weight is not None:
            assert weight.ndim == 1
            assert p_i.shape[2] == len(weight)

        self.p_i = p_i
        self.p_ij = p_ij
        self.weight = weight

        # covariance matrix of shap vectors
        # shape: (n_outputs, n_features, n_features)
        self.cov_p_i_p_j = cov_p_i_p_j = cov(p_i, weight)

        # var(p[i])
        # variances of SHAP vectors
        # shape: (n_outputs, n_features, 1)
        # i.e. adding a second, empty feature dimension to enable correct broadcasting
        self.var_p_i = diagonal(cov_p_i_p_j)[:, :, np.newaxis]


class ShapValueContext(ShapContext):
    """
    Contextual data for global SHAP calculations based on SHAP values.
    """

    def __init__(self, shap_calculator: ShapCalculator, split_id: int) -> None:
        shap_values: pd.DataFrame = shap_calculator.get_shap_values(
            aggregation=None
        ).xs(split_id, level=0)

        def _p_i() -> np.ndarray:
            n_outputs: int = len(shap_calculator.output_names_)
            n_features: int = len(shap_calculator.feature_index_)
            n_observations: int = len(shap_values)

            # p[i] = p_i
            # shape: (n_outputs, n_features, n_observations)
            # the vector of shap values for every output and feature
            return ensure_last_axis_is_fast(
                np.transpose(
                    shap_values.values.reshape((n_observations, n_outputs, n_features)),
                    axes=(1, 2, 0),
                )
            )

        def _weight() -> Optional[np.ndarray]:
            # weights
            # shape: (n_observations)
            # return a 1d array of weights that aligns with the observations axis of the
            # SHAP values tensor (axis 1)
            _weight_sr = shap_calculator.sample_.weight
            if _weight_sr is not None:
                return _weight_sr.loc[shap_values.index.get_level_values(-1)].values
            else:
                return None

        super().__init__(p_i=_p_i(), p_ij=None, weight=_weight())


class ShapInteractionValueContext(ShapContext):
    """
    Contextual data for global SHAP calculations based on SHAP interaction values.
    """

    def __init__(self, shap_calculator: ShapCalculator, split_id: int) -> None:
        shap_values: pd.DataFrame = shap_calculator.get_shap_interaction_values(
            aggregation=None
        ).xs(split_id, level=0)

        n_features: int = len(shap_calculator.feature_index_)
        n_outputs: int = len(shap_calculator.output_names_)
        n_observations: int = len(shap_values) // n_features

        assert shap_values.shape == (
            n_observations * n_features,
            n_outputs * n_features,
        )

        self.matrix_shape = (n_outputs, n_features, n_features)

        # weights
        # shape: (n_observations)
        # return a 1d array of weights that aligns with the observations axis of the
        # SHAP values tensor (axis 1)
        weight: Optional[np.ndarray]
        _weight_sr = shap_calculator.sample_.weight
        if _weight_sr is not None:
            _observation_indices = shap_values.index.get_level_values(
                -2
            ).values.reshape((n_observations, n_features))[:, 0]
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
        super().__init__(
            p_i=ensure_last_axis_is_fast(p_ij.sum(axis=2)),
            p_ij=ensure_last_axis_is_fast(
                self.__get_orthogonalized_interaction_vectors(p_ij=p_ij, weight=weight)
            ),
            weight=weight,
        )

    @staticmethod
    def __get_orthogonalized_interaction_vectors(
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

        fill_diagonal(adjustment_factors_ij, np.nan)

        delta_ij = (
            adjustment_factors_ij[:, :, :, np.newaxis] * p_ii[:, :, np.newaxis, :]
        )
        return p_ij - delta_ij - transpose(delta_ij, ndim=4)


__tracker.validate()
