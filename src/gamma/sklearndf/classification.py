"""
Data frame versions of all sklearn regressors
"""
import logging
from typing import *

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

from gamma.sklearndf import ClassifierDF, T_Classifier
from gamma.sklearndf._wrapper import ClassifierWrapperDF, df_estimator

log = logging.getLogger(__name__)


# [sym for sym in dir(classification) if sym.endswith("DF")]
__all__ = [
    "AdaBoostClassifierDF",
    "BaggingClassifierDF",
    "BernoulliNBDF",
    "CalibratedClassifierCVDF",
    "ClassifierChainDF",
    "ClassifierDF",
    "ClassifierWrapperDF",
    "ComplementNBDF",
    "DecisionTreeClassifierDF",
    "DummyClassifierDF",
    "ExtraTreeClassifierDF",
    "ExtraTreesClassifierDF",
    "GaussianNBDF",
    "GaussianProcessClassifierDF",
    "GradientBoostingClassifierDF",
    "KNeighborsClassifierDF",
    "LabelPropagationDF",
    "LabelSpreadingDF",
    "LinearDiscriminantAnalysisDF",
    "LinearSVCDF",
    "LogisticRegressionCVDF",
    "LogisticRegressionDF",
    "MLPClassifierDF",
    "MultiOutputClassifierDF",
    "MultinomialNBDF",
    "NearestCentroidDF",
    "NuSVCDF",
    "OneVsOneClassifierDF",
    "OneVsRestClassifierDF",
    "OutputCodeClassifierDF",
    "PassiveAggressiveClassifierDF",
    "PerceptronDF",
    "QuadraticDiscriminantAnalysisDF",
    "RadiusNeighborsClassifierDF",
    "RandomForestClassifierDF",
    "RidgeClassifierCVDF",
    "RidgeClassifierDF",
    "SGDClassifierDF",
    "SVCDF",
    "VotingClassifierDF",
]

#
# decorator for wrapping the sklearn classifier classes
#


def _df_classifier(
    delegate_classifier: Type[T_Classifier]
) -> Type[ClassifierWrapperDF[T_Classifier]]:
    return cast(
        Type[ClassifierWrapperDF[T_Classifier]],
        df_estimator(
            delegate_estimator=delegate_classifier,
            df_estimator_type=ClassifierWrapperDF,
        ),
    )


#
# neighbors
#


# noinspection PyAbstractClass
@_df_classifier
class NearestCentroidDF(NearestCentroid, ClassifierDF):
    """
    Wraps :class:`sklearn.neighbors.nearest_centroid.NearestCentroid`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class KNeighborsClassifierDF(KNeighborsClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.neighbors.classification.KNeighborsClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class RadiusNeighborsClassifierDF(RadiusNeighborsClassifier, ClassifierDF):
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
class VotingClassifierDF(VotingClassifier, ClassifierDF):
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
class RandomForestClassifierDF(RandomForestClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.ensemble.forest.RandomForestClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ExtraTreesClassifierDF(ExtraTreesClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.ensemble.forest.ExtraTreesClassifier`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class GradientBoostingClassifierDF(GradientBoostingClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.ensemble.gradient_boosting.GradientBoostingClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class AdaBoostClassifierDF(AdaBoostClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.ensemble.weight_boosting.AdaBoostClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class BaggingClassifierDF(BaggingClassifier, ClassifierDF):
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
class DecisionTreeClassifierDF(DecisionTreeClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.tree.tree.DecisionTreeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ExtraTreeClassifierDF(ExtraTreeClassifier, ClassifierDF):
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
class QuadraticDiscriminantAnalysisDF(QuadraticDiscriminantAnalysis, ClassifierDF):
    """
    Wraps :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LinearDiscriminantAnalysisDF(LinearDiscriminantAnalysis, ClassifierDF):
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
class GaussianNBDF(GaussianNB, ClassifierDF):
    """
    Wraps :class:`sklearn.naive_bayes.GaussianNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class MultinomialNBDF(MultinomialNB, ClassifierDF):
    """
    Wraps :class:`sklearn.naive_bayes.MultinomialNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ComplementNBDF(ComplementNB, ClassifierDF):
    """
    Wraps :class:`sklearn.naive_bayes.ComplementNB`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class BernoulliNBDF(BernoulliNB, ClassifierDF):
    """
    Wraps :class:`sklearn.naive_bayes.BernoulliNB`; accepts and returns data frames.
    """

    pass


#
# calibration
#


class CalibratedClassifierCVDF(ClassifierWrapperDF[CalibratedClassifierCV]):
    """
    Wraps :class:`sklearn.calibration.CalibratedClassifierCV`; accepts and returns data
    frames.
    """

    def __init__(self, base_estimator: ClassifierDF, **kwargs):
        super().__init__(base_estimator=base_estimator.delegate_estimator, **kwargs)
        self.base_estimator_df = base_estimator

    @classmethod
    def _make_delegate_estimator(cls, **kwargs) -> CalibratedClassifierCV:
        return CalibratedClassifierCV(**kwargs)


#
# SVM
#


# noinspection PyAbstractClass
@_df_classifier
class SVCDF(SVC, ClassifierDF):
    """
    Wraps :class:`sklearn.svm.classes.SVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class NuSVCDF(NuSVC, ClassifierDF):
    """
    Wraps :class:`sklearn.svm.classes.NuSVC`; accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LinearSVCDF(LinearSVC, ClassifierDF):
    """
    Wraps :class:`sklearn.svm.classes.LinearSVC`; accepts and returns data frames.
    """

    pass


#
# dummy
#


# noinspection PyAbstractClass
@_df_classifier
class DummyClassifierDF(DummyClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.dummy.DummyClassifier`; accepts and returns data frames.
    """

    pass


#
# gaussian process
#


# noinspection PyAbstractClass
@_df_classifier
class GaussianProcessClassifierDF(GaussianProcessClassifier, ClassifierDF):
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
class LogisticRegressionDF(LogisticRegression, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegression`; accepts and returns
    data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LogisticRegressionCVDF(LogisticRegressionCV, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LogisticRegressionCVDF(LogisticRegressionCV, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.logistic.LogisticRegressionCV`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class PassiveAggressiveClassifierDF(PassiveAggressiveClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier`;
    accepts and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class PerceptronDF(Perceptron, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.perceptron.Perceptron`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class SGDClassifierDF(SGDClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.stochastic_gradient.SGDClassifier`; accepts and
    returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class RidgeClassifierDF(RidgeClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.linear_model.ridge.RidgeClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class RidgeClassifierCVDF(RidgeClassifierCV, ClassifierDF):
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
class LabelPropagationDF(LabelPropagation, ClassifierDF):
    """
    Wraps :class:`sklearn.semi_supervised.label_propagation.LabelPropagation`; accepts
    and returns data frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class LabelSpreadingDF(LabelSpreading, ClassifierDF):
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
class OneVsRestClassifierDF(OneVsRestClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.multiclass.OneVsRestClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class OneVsOneClassifierDF(OneVsOneClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.multiclass.OneVsOneClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class OutputCodeClassifierDF(OutputCodeClassifier, ClassifierDF):
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
class MultiOutputClassifierDF(MultiOutputClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.multioutput.MultiOutputClassifier`; accepts and returns data
    frames.
    """

    pass


# noinspection PyAbstractClass
@_df_classifier
class ClassifierChainDF(ClassifierChain, ClassifierDF):
    """
    Wraps :class:`sklearn.multioutput.ClassifierChain`; accepts and returns data frames.
    """

    pass


#
# neural network
#


# noinspection PyAbstractClass
@_df_classifier
class MLPClassifierDF(MLPClassifier, ClassifierDF):
    """
    Wraps :class:`sklearn.neural_network.multilayer_perceptron.MLPClassifier`; accepts
    and returns data frames.
    """

    pass
