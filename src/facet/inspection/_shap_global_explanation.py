"""
Projection of SHAP contribution scores (i.e, SHAP importance) of all possible
pairings of features onto the SHAP importance vector in partitions of for synergy,
redundancy, and independence.
"""
import logging
from abc import ABCMeta, abstractmethod
from typing import List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin

from ._shap import ShapCalculator

log = logging.getLogger(__name__)

__all__ = [
    "AffinityMatrices",
    "ShapGlobalExplainer",
    "ShapInteractionGlobalExplainer",
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


class AffinityMatrices:
    """
    Stores all variations of a feature affinity matrix.
    """

    def __init__(self, affinity_rel_ij: np.ndarray, std_p_i: np.ndarray) -> None:
        """
        :param affinity_rel_ij: the affinity matrix from which to create all variations,
            shaped `(n_outputs, n_features, n_features)`
        :param std_p_i: feature importance for all outputs and features,
            shaped `(n_outputs, n_features)`
        """
        assert affinity_rel_ij.ndim == 3
        assert std_p_i.ndim == 3
        assert affinity_rel_ij.shape[:2] == std_p_i.shape[:2]
        assert affinity_rel_ij.shape[1] == affinity_rel_ij.shape[2]
        assert std_p_i.shape[2] == 1

        affinity_abs_ij = std_p_i * affinity_rel_ij
        affinity_abs_sym_ij_2x = affinity_abs_ij + transpose(affinity_abs_ij)
        affinity_rel_sym_ij = affinity_abs_sym_ij_2x / (std_p_i + transpose(std_p_i))
        fill_diagonal(affinity_rel_sym_ij, 1.0)
        self._matrices = (
            (affinity_rel_ij, affinity_abs_ij),
            (affinity_rel_sym_ij, affinity_abs_sym_ij_2x / 2),
        )

    def get_matrix(self, symmetrical: bool, absolute: bool):
        """
        Get the matrix matching the given criteria.
        :param symmetrical: if ``True``, get the symmetrical version of the matrix
        :param absolute: if ``True``, get the absolute version of the matrix
        :return: the affinity matrix
        """
        return self._matrices[bool(symmetrical)][bool(absolute)]


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

    def fit(self: T_Self, shap_calculator: ShapCalculator, **fit_params) -> T_Self:
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
    def association(self, absolute: bool, symmetrical: bool) -> np.ndarray:
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
    def synergy(self, symmetrical: bool, absolute: bool) -> np.ndarray:
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

    @abstractmethod
    def redundancy(self, symmetrical: bool, absolute: bool) -> np.ndarray:
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


def transpose(m: np.ndarray) -> np.ndarray:
    """
    Transpose the `feature x feature` matrix for each output.

    Supports matrices with identical values per row, represented as a broadcastable
    `numpy` array of shape `(n_features, 1)`.

    :param m: array of shape `(n_outputs, n_features, n_features)`
        or shape `(n_outputs, n_features, n_features, n_observations)`
        or shape `(n_outputs, n_features, 1)`
        or shape `(n_outputs, n_features, 1, n_observations)`
    :return: array of same shape as arg ``m``, with both feature axes swapped
    """
    assert 3 <= m.ndim <= 4
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


__tracker.validate()
