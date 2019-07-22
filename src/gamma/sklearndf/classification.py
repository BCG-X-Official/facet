"""
Data frame versions of all sklearn regressors
"""
import logging
from typing import *

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    RidgeClassifierCV,
    SGDClassifier,
)
from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from gamma.sklearndf import _DataFramePredictor, DataFrameClassifier
from gamma.sklearndf._wrapper import DataFrameClassifierWrapper, df_estimator

log = logging.getLogger(__name__)

#
# decorator for wrapping the sklearn classifier classes
#


def _df_classifier(base_classifier: Type[BaseEstimator]):
    return df_estimator(
        base_estimator=base_classifier, df_estimator_type=DataFrameClassifierWrapper
    )


#
# type hinting
#

# noinspection PyAbstractClass
class _DataFrameClassifier(DataFrameClassifier, _DataFramePredictor):
    """Dummy data frame regressor class, for type hinting only."""

    pass


#
# neighbors
#


# noinspection PyAbstractClass
@_df_classifier
class NearestCentroidDF(NearestCentroid, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.neighbors.nearest_centroid.NearestCentroid`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class KNeighborsClassifierDF(KNeighborsClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.KNeighborsClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class RadiusNeighborsClassifierDF(RadiusNeighborsClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.neighbors.classification.RadiusNeighborsClassifier`; accepts
    and returns data frames.
    """

    pass


#
# voting
#


# noinspection PyAbstractClass
@_df_classifier
class VotingClassifierDF(VotingClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.ensemble.voting.VotingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# ensemble
#


# noinspection PyAbstractClass
@_df_classifier
class RandomForestClassifierDF(RandomForestClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ExtraTreesClassifierDF(ExtraTreesClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class GradientBoostingClassifierDF(GradientBoostingClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class AdaBoostClassifierDF(AdaBoostClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class BaggingClassifierDF(BaggingClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.ensemble.bagging.BaggingClassifier`; accepts and returns data
    frames.
    """

    pass


#
# tree
#


# noinspection PyAbstractClass
@_df_classifier
class DecisionTreeClassifierDF(DecisionTreeClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ExtraTreeClassifierDF(ExtraTreeClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.tree.tree.ExtraTreeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# discriminant analysis
#


# noinspection PyAbstractClass
@_df_classifier
class QuadraticDiscriminantAnalysisDF(
    QuadraticDiscriminantAnalysis, DataFrameClassifier
):
    """
    Wraps :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LinearDiscriminantAnalysisDF(LinearDiscriminantAnalysis, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`; accepts and
    returns data frames.
    """

    pass


#
# naive bayes
#


# noinspection PyAbstractClass
@_df_classifier
class GaussianNBDF(GaussianNB, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.naive_bayes.GaussianNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class MultinomialNBDF(MultinomialNB, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.naive_bayes.MultinomialNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ComplementNBDF(ComplementNB, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.naive_bayes.ComplementNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class BernoulliNBDF(BernoulliNB, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.naive_bayes.BernoulliNB`; accepts and returns data frames.
    """

    pass


#
# calibration
#


# noinspection PyAbstractClass
@_df_classifier
class CalibratedClassifierCVDF(CalibratedClassifierCV, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.calibration.CalibratedClassifierCV`; accepts and returns data
    frames.
    """

    pass


#
# SVM
#


# noinspection PyAbstractClass
@_df_classifier
class SVCDF(SVC, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.svm.classes.SVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class NuSVCDF(NuSVC, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.svm.classes.NuSVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LinearSVCDF(LinearSVC, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVC`; accepts and returns data frames.
    """

    pass


#
# dummy
#


# noinspection PyAbstractClass
@_df_classifier
class DummyClassifierDF(DummyClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.dummy.DummyClassifier`; accepts and returns data frames.
    """

    pass


#
# gaussian process
#


# noinspection PyAbstractClass
@_df_classifier
class GaussianProcessClassifierDF(GaussianProcessClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.gaussian_process.gpc.GaussianProcessClassifier`; accepts and
    returns data frames.
    """

    pass


#
# linear model
#


# noinspection PyAbstractClass
@_df_classifier
class LogisticRegressionDF(LogisticRegression, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LogisticRegressionCVDF(LogisticRegressionCV, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LogisticRegressionCVDF(LogisticRegressionCV, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class PassiveAggressiveClassifierDF(PassiveAggressiveClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class PerceptronDF(Perceptron, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.perceptron.Perceptron`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class SGDClassifierDF(SGDClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class RidgeClassifierDF(RidgeClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class RidgeClassifierCVDF(RidgeClassifierCV, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifierCV`; accepts and returns
    data frames.
    """

    pass


#
# semi-supervised
#


# noinspection PyAbstractClass
@_df_classifier
class LabelPropagationDF(LabelPropagation, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelPropagation`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LabelSpreadingDF(LabelSpreading, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelSpreading`; accepts and
    returns data frames.
    """

    pass


#
# multi-class
#


# noinspection PyAbstractClass
@_df_classifier
class OneVsRestClassifierDF(OneVsRestClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsRestClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class OneVsOneClassifierDF(OneVsOneClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.multiclass.OneVsOneClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class OutputCodeClassifierDF(OutputCodeClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.multiclass.OutputCodeClassifier`; accepts and returns data
    frames.
    """

    pass


#
# multi-output
#


# noinspection PyAbstractClass
@_df_classifier
class MultiOutputClassifierDF(MultiOutputClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ClassifierChainDF(ClassifierChain, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.multioutput.ClassifierChain`; accepts and returns data frames.
    """

    pass


#
# neural network
#


# noinspection PyAbstractClass
@_df_classifier
class MLPClassifierDF(MLPClassifier, _DataFrameClassifier):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPClassifier`; accepts
    and returns data frames.
    """

    pass
