"""
Data frame versions of all sklearn regressors
"""
import logging
from typing import Type

from lightgbm.sklearn import LGBMRegressor
from sklearn.base import BaseEstimator
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

from yieldengine.df import df_estimator
from yieldengine.df.predict import DataFrameRegressor, DataFrameRegressorWrapper
from yieldengine.prediction import _DataFramePredictor

log = logging.getLogger(__name__)


#
# decorator for wrapping the sklearn regressor classes
#


def _df_regressor(base_regressor: Type[BaseEstimator]):
    return df_estimator(
        base_estimator=base_regressor, df_estimator_type=DataFrameRegressorWrapper
    )


#
# type hinting
#

# noinspection PyAbstractClass
class _DataFrameRegressor(DataFrameRegressor, _DataFramePredictor):
    """Dummy data frame regressor class, for type hinting only."""

    pass


#
# SVM
#


# noinspection PyAbstractClass


@_df_regressor
class LinearSVRDF(LinearSVR, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVR`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class SVRDF(SVR, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.svm.classes.SVR`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class NuSVRDF(NuSVR, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.svm.classes.NuSVR`; accepts and returns data frames.
    """

    pass


#
# dummy
#


# noinspection PyAbstractClass
@_df_regressor
class DummyRegressorDF(DummyRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.dummy.DummyRegressor`; accepts and returns data frames.
    """

    pass


#
# multi-output
#


# noinspection PyAbstractClass
@_df_regressor
class MultiOutputRegressorDF(MultiOutputRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RegressorChainDF(RegressorChain, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.multioutput.RegressorChain`; accepts and returns data frames.
    """

    pass


#
# neighbors
#


# noinspection PyAbstractClass
@_df_regressor
class KNeighborsRegressorDF(KNeighborsRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.neighbors.regression.KNeighborsRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RadiusNeighborsRegressorDF(RadiusNeighborsRegressor, _DataFrameRegressor):
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
class MLPRegressorDF(MLPRegressor, _DataFrameRegressor):
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
class LinearRegressionDF(LinearRegression, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.base.LinearRegression`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RidgeDF(Ridge, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.ridge.Ridge`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RidgeCVDF(RidgeCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeCV`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class SGDRegressorDF(SGDRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDRegressor`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class HuberRegressorDF(HuberRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.huber.HuberRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class TheilSenRegressorDF(TheilSenRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.theil_sen.TheilSenRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class BayesianRidgeDF(BayesianRidge, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.bayes.BayesianRidge`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ARDRegressionDF(ARDRegression, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.bayes.ARDRegression`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class OrthogonalMatchingPursuitDF(OrthogonalMatchingPursuit, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.omp.OrthogonalMatchingPursuit`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class OrthogonalMatchingPursuitCVDF(OrthogonalMatchingPursuitCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.omp.OrthogonalMatchingPursuitCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RANSACRegressorDF(RANSACRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.ransac.RANSACRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ElasticNetDF(ElasticNet, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.ElasticNet`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoCVDF(LassoCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.LassoCV`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ElasticNetCVDF(ElasticNetCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.ElasticNetCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskElasticNetCVDF(MultiTaskElasticNetCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskElasticNetCV`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskLassoCVDF(MultiTaskLassoCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskLassoCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskElasticNetDF(MultiTaskElasticNet, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskElasticNet`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class MultiTaskLassoDF(MultiTaskLasso, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.MultiTaskLasso`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoDF(Lasso, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.coordinate_descent.Lasso`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class PassiveAggressiveRegressorDF(PassiveAggressiveRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveRegressor`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LarsDF(Lars, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.least_angle.Lars`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoLarsDF(LassoLars, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLars`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoLarsICDF(LassoLarsIC, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LassoLarsIC`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LarsCVDF(LarsCV, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.linear_model.least_angle.LarsCV`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class LassoLarsCVDF(LassoLarsCV, _DataFrameRegressor):
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
class BaggingRegressorDF(BaggingRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class VotingRegressorDF(VotingRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
# noinspection PyAbstractClass
@_df_regressor
class GradientBoostingRegressorDF(GradientBoostingRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingRegressor`; accepts
    and returns data frames.
    """


# noinspection PyAbstractClass
@_df_regressor
class AdaBoostRegressorDF(AdaBoostRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostRegressor`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class RandomForestRegressorDF(RandomForestRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestRegressor`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ExtraTreesRegressorDF(ExtraTreesRegressor, _DataFrameRegressor):
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
class GaussianProcessRegressorDF(GaussianProcessRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.gaussian_process.gpr.GaussianProcessRegressor`; accepts and
    returns data frames.
    """

    pass


#
# isotonic
#


# noinspection PyAbstractClass
@_df_regressor
class IsotonicRegressionDF(IsotonicRegression, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.isotonic.IsotonicRegression`; accepts and returns data frames.
    """

    pass


#
# compose
#


# noinspection PyAbstractClass
@_df_regressor
class TransformedTargetRegressorDF(TransformedTargetRegressor, _DataFrameRegressor):
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
class KernelRidgeDF(KernelRidge, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.kernel_ridge.KernelRidge`; accepts and returns data frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@_df_regressor
class DecisionTreeRegressorDF(DecisionTreeRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeRegressor`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class ExtraTreeRegressorDF(ExtraTreeRegressor, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeRegressor`; accepts and returns data
    frames.
    """

    pass


#
# cross_decomposition
#


# noinspection PyAbstractClass
@_df_regressor
class CCADF(CCA, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.cross_decomposition.cca_.CCA`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class PLSRegressionDF(PLSRegression, _DataFrameRegressor):
    """
    Wraps :class:`sklearn.cross_decomposition.pls_.PLSRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_regressor
class PLSCanonicalDF(PLSCanonical, _DataFrameRegressor):
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
class LGBMRegressorDF(LGBMRegressor, _DataFrameRegressor):
    """
    Wraps :class:`lightgbm.sklearn.LGBMRegressor`; accepts and returns data frames.
    """

    pass
