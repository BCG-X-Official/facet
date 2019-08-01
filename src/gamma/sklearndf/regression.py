"""
Data frame versions of all sklearn regressors
"""
import logging
from typing import *

from lightgbm.sklearn import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    ElasticNetCV,
    HuberRegressor,
    Lars,
    LarsCV,
    Lasso,
    LassoCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    LinearRegression,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    RidgeCV,
    TheilSenRegressor,
)
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from gamma.sklearndf import RegressorDF, T_Regressor, TransformerDF
from gamma.sklearndf._wrapper import df_estimator, RegressorWrapperDF
from gamma.sklearndf.transformation import ColumnPreservingTransformerWrapperDF

log = logging.getLogger(__name__)

# [sym for sym in dir(regression) if sym.endswith("DF")]
__all__ = [
    "ARDRegressionDF",
    "AdaBoostRegressorDF",
    "BaggingRegressorDF",
    "BayesianRidgeDF",
    "CCADF",
    "DecisionTreeRegressorDF",
    "DummyRegressorDF",
    "ElasticNetCVDF",
    "ElasticNetDF",
    "ExtraTreeRegressorDF",
    "ExtraTreesRegressorDF",
    "GaussianProcessRegressorDF",
    "GradientBoostingRegressorDF",
    "HuberRegressorDF",
    "IsotonicRegressionDF",
    "KNeighborsRegressorDF",
    "KernelRidgeDF",
    "LGBMRegressorDF",
    "LarsCVDF",
    "LarsDF",
    "LassoCVDF",
    "LassoDF",
    "LassoLarsCVDF",
    "LassoLarsDF",
    "LassoLarsICDF",
    "LinearRegressionDF",
    "LinearSVRDF",
    "MLPRegressorDF",
    "MultiOutputRegressorDF",
    "MultiTaskElasticNetCVDF",
    "MultiTaskElasticNetDF",
    "MultiTaskLassoCVDF",
    "MultiTaskLassoDF",
    "NuSVRDF",
    "OrthogonalMatchingPursuitCVDF",
    "OrthogonalMatchingPursuitDF",
    "PLSCanonicalDF",
    "PLSRegressionDF",
    "PassiveAggressiveRegressorDF",
    "RANSACRegressorDF",
    "RadiusNeighborsRegressorDF",
    "RandomForestRegressorDF",
    "RegressorChainDF",
    "RegressorDF",
    "RegressorWrapperDF",
    "RidgeCVDF",
    "RidgeDF",
    "SGDRegressorDF",
    "SVRDF",
    "TheilSenRegressorDF",
    "TransformedTargetRegressorDF",
    "TransformerDF",
    "VotingRegressorDF",
]

#
# decorator for wrapping the sklearn regressor classes
#


def _df_regressor(
    delegate_regressor: Type[T_Regressor]
) -> Type[RegressorWrapperDF[T_Regressor]]:
    return cast(
        Type[RegressorWrapperDF[T_Regressor]],
        df_estimator(
            delegate_estimator=delegate_regressor, df_wrapper_type=RegressorWrapperDF
        ),
    )


class _RegressorTransformerWrapperDF(
    RegressorWrapperDF[T_Regressor],
    ColumnPreservingTransformerWrapperDF[T_Regressor],
    Generic[T_Regressor],
):
    """
    Wraps a combined regressor and constant column transformer
    """

    pass


def _df_regressor_transformer(
    delegate_regressor_transformer: Type[T_Regressor]
) -> Type[_RegressorTransformerWrapperDF[T_Regressor]]:
    return cast(
        Type[_RegressorTransformerWrapperDF[T_Regressor]],
        df_estimator(
            delegate_estimator=delegate_regressor_transformer,
            df_wrapper_type=_RegressorTransformerWrapperDF,
        ),
    )


#
# SVM
#


# noinspection PyAbstractClass


@_df_regressor
class LinearSVRDF(LinearSVR, RegressorDF):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVR`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class SVRDF(SVR, RegressorDF):
    """
    Wraps :class:`sklearn.svm.classes.SVR`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class NuSVRDF(NuSVR, RegressorDF):
    """
    Wraps :class:`sklearn.svm.classes.NuSVR`; accepts and returns data frames.
    """

    pass


#
# dummy
#


# noinspection PyAbstractClass
@_df_regressor
class DummyRegressorDF(DummyRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.dummy.DummyRegressor`; accepts and returns data frames.
    """

    pass


#
# multi-output
#


# noinspection PyAbstractClass
@_df_regressor
class MultiOutputRegressorDF(MultiOutputRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RegressorChainDF(RegressorChain, RegressorDF):
    """
    Wraps :class:`sklearn.multioutput.RegressorChain`; accepts and returns data frames.
    """

    pass


#
# neighbors
#


# noinspection PyAbstractClass
@_df_regressor
class KNeighborsRegressorDF(KNeighborsRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.neighbors.regression.KNeighborsRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RadiusNeighborsRegressorDF(RadiusNeighborsRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.neighbors.regression.RadiusNeighborsRegressor`; accepts and
    returns data frames.
    """

    pass


#
# neural_network
#


# noinspection PyAbstractClass
@_df_regressor
class MLPRegressorDF(MLPRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPRegressor`; accepts
    and returns data frames.
    """

    pass


#
# linear_model
#


# noinspection PyAbstractClass
@_df_regressor
class LinearRegressionDF(LinearRegression, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.base.LinearRegression`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RidgeDF(Ridge, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.ridge.Ridge`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RidgeCVDF(RidgeCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeCV`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class SGDRegressorDF(SGDRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDRegressor`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class HuberRegressorDF(HuberRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.huber.HuberRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class TheilSenRegressorDF(TheilSenRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.theil_sen.TheilSenRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class BayesianRidgeDF(BayesianRidge, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.bayes.BayesianRidge`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ARDRegressionDF(ARDRegression, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.bayes.ARDRegression`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class OrthogonalMatchingPursuitDF(OrthogonalMatchingPursuit, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.omp.OrthogonalMatchingPursuit`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class OrthogonalMatchingPursuitCVDF(OrthogonalMatchingPursuitCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.omp.OrthogonalMatchingPursuitCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RANSACRegressorDF(RANSACRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.ransac.RANSACRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ElasticNetDF(ElasticNet, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.ElasticNet`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoCVDF(LassoCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.LassoCV`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ElasticNetCVDF(ElasticNetCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.ElasticNetCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskElasticNetCVDF(MultiTaskElasticNetCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskElasticNetCV`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskLassoCVDF(MultiTaskLassoCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskLassoCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskElasticNetDF(MultiTaskElasticNet, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskElasticNet`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskLassoDF(MultiTaskLasso, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskLasso`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoDF(Lasso, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.Lasso`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class PassiveAggressiveRegressorDF(PassiveAggressiveRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveRegressor`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LarsDF(Lars, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.least_angle.Lars`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoLarsDF(LassoLars, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLars`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoLarsICDF(LassoLarsIC, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLarsIC`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LarsCVDF(LarsCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LarsCV`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoLarsCVDF(LassoLarsCV, RegressorDF):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLarsCV`; accepts and returns
    data frames.
    """

    pass


#
# ensemble
#


# noinspection PyAbstractClass
@_df_regressor
class BaggingRegressorDF(BaggingRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class VotingRegressorDF(VotingRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
# noinspection PyAbstractClass
@_df_regressor
class GradientBoostingRegressorDF(GradientBoostingRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingRegressor`; accepts
    and returns data frames.
    """


# noinspection PyAbstractClass
@_df_regressor
class AdaBoostRegressorDF(AdaBoostRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostRegressor`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RandomForestRegressorDF(RandomForestRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ExtraTreesRegressorDF(ExtraTreesRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesRegressor`; accepts and returns data
    frames.
    """

    pass


#
# gaussian_process
#


# noinspection PyAbstractClass
@_df_regressor
class GaussianProcessRegressorDF(GaussianProcessRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.gaussian_process.gpr.GaussianProcessRegressor`; accepts and
    returns data frames.
    """

    pass


#
# isotonic
#


# noinspection PyAbstractClass
@_df_regressor_transformer
class IsotonicRegressionDF(IsotonicRegression, RegressorDF, TransformerDF):
    """
    Wraps :class:`sklearn.isotonic.IsotonicRegression`; accepts and returns data frames.
    """

    pass


#
# compose
#


# noinspection PyAbstractClass
@_df_regressor
class TransformedTargetRegressorDF(TransformedTargetRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.compose._target.TransformedTargetRegressor`; accepts and
    returns data frames.
    """

    pass


#
# kernel_ridge
#


# noinspection PyAbstractClass
@_df_regressor
class KernelRidgeDF(KernelRidge, RegressorDF):
    """
    Wraps :class:`sklearn.kernel_ridge.KernelRidge`; accepts and returns data frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@_df_regressor
class DecisionTreeRegressorDF(DecisionTreeRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ExtraTreeRegressorDF(ExtraTreeRegressor, RegressorDF):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeRegressor`; accepts and returns data
    frames.
    """

    pass


#
# cross_decomposition
#


# noinspection PyAbstractClass
@_df_regressor_transformer
class CCADF(CCA, RegressorDF, TransformerDF):
    """
    Wraps :class:`sklearn.cross_decomposition.cca_.CCA`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor_transformer
class PLSRegressionDF(PLSRegression, RegressorDF, TransformerDF):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor_transformer
class PLSCanonicalDF(PLSCanonical, RegressorDF, TransformerDF):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSCanonical`; accepts and returns
    data frames.
    """

    pass


#
# lightgbm
#

# noinspection PyAbstractClass
@_df_regressor
class LGBMRegressorDF(LGBMRegressor, RegressorDF):
    """
    Wraps :class:`lightgbm.sklearn.LGBMRegressor`; accepts and returns data frames.
    """

    pass
