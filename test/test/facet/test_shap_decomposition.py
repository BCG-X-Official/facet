"""
Test shap decomposition calculations
"""
import logging
from typing import Set, Union

import numpy as np

from facet.inspection import LearnerInspector

log = logging.getLogger(__name__)


def test_feature_affinity_matrices(
    preprocessed_feature_names: Set[str], regressor_inspector: LearnerInspector
) -> None:
    # feature affinity matrices (feature dependencies)
    # check that dimensions of pairwise feature matrices are equal to # of features,
    # and value ranges:
    for matrix, matrix_name in zip(
        (
            regressor_inspector.feature_association_matrix(),
            regressor_inspector.feature_synergy_matrix(),
            regressor_inspector.feature_redundancy_matrix(),
        ),
        ("association", "synergy", "redundancy"),
    ):
        matrix_full_name = f"feature {matrix_name} matrix"
        n_features = len(preprocessed_feature_names)
        assert matrix.values.shape[0] == n_features, f"rows in {matrix_full_name}"
        assert matrix.values.shape[1] == n_features, f"columns in {matrix_full_name}"
        assert (
            set(matrix.names[0]) == preprocessed_feature_names
        ), f"row names in {matrix_full_name}"
        assert (
            set(matrix.names[1]) == preprocessed_feature_names
        ), f"column names in {matrix_full_name}"

        # check values
        assert (
            np.nanmin(matrix.values) >= 0.0 and np.nanmax(matrix.values) <= 1.0
        ), f"Values of [0.0, 1.0] in {matrix_full_name}"


#
# auxiliary functions
#


def cov(a: np.ndarray, b: np.ndarray) -> float:
    """
    covariance, assuming a population mean of 0
    :param a: array of floats
    :param b: array of floats
    :return: covariance of a and b
    """
    return (a * b).mean()


def var(a: np.ndarray) -> float:
    """
    variance, assuming a population mean of 0
    :param a: array of floats
    :return: variance of a
    """
    return cov(a, a)


def std(a: np.ndarray) -> float:
    """
    standard deviation, assuming a population mean of 0
    :param a: array of floats
    :return: standard deviation of a
    """
    return np.sqrt(var(a))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """
    pearson correlation, assuming a population mean of 0
    :param a: array of floats
    :param b: array of floats
    :return: pearson correlation of a and b
    """
    return cov(a, b) / np.sqrt(var(a) * var(b))


def print_list(*args, percentage: bool = False, **kwargs):
    """
    print all arguments, including their names
    :param args: the arguments to print (as their names, print integers indicating \
        the position)
    :param percentage: if `true`, print all arguments as % values
    :param kwargs: the named arguments to print
    :return:
    """

    def _prt(_value, _name: Union[str, int]):
        if percentage:
            _value *= 100
        print(f"{_name}: {_value:.4g}{'%' if percentage else ''}")

    for name, arg in enumerate(args):
        _prt(arg, _name=name)
    for name, arg in kwargs.items():
        _prt(arg, _name=name)
