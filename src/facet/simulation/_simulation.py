"""
Core implementation of :mod:`facet.simulation`
"""

import logging
from typing import Any, Optional, Tuple, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from sklearndf import ClassifierDF, RegressorDF

from ..data import Sample
from ..data.partition import Partitioner
from ._result import UnivariateSimulationResult
from .base import BaseUnivariateSimulator, UnivariateRegressionSimulator

log = logging.getLogger(__name__)

__all__ = [
    "UnivariateProbabilitySimulator",
    "UnivariateTargetSimulator",
    "UnivariateUpliftSimulator",
]


#
# Type variables
#

T_Values = TypeVar("T_Values", bound=np.generic)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="[see superclass]")
class UnivariateProbabilitySimulator(BaseUnivariateSimulator[ClassifierDF]):
    """
    Univariate simulation of positive class probabilities based on a binary classifier.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioner`
    object.

    For each value `v[j]` of the partitioning, a :class:`.Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then the classifier is used to predict the positive class probabilities for all
    observations, and the mean probability across all observations is calculated
    for each classifier and value `v[j]`,
    along with the standard error of the mean as a basis of obtaining confidence
    intervals.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.

    Also, care should be taken to re-calibrate classifiers trained on weighted samples
    as the weighted samples will impact predicted class probabilities.
    """

    # defined in superclass, repeated here for Sphinx
    n_jobs: Optional[int]

    # defined in superclass, repeated here for Sphinx
    shared_memory: Optional[bool]

    # defined in superclass, repeated here for Sphinx
    pre_dispatch: Optional[Union[str, int]]

    # defined in superclass, repeated here for Sphinx
    verbose: Optional[int]

    # defined in superclass, repeated here for Sphinx
    model: ClassifierDF

    # defined in superclass, repeated here for Sphinx
    sample: Sample

    # defined in superclass, repeated here for Sphinx
    confidence_level: float

    @property
    def output_unit(self) -> str:
        """[see superclass]"""
        return f"probability({self._positive_class()})"

    def expected_output(self) -> float:
        """
        Calculate the actual observed frequency of the positive class as the baseline
        of the simulation.

        :return: observed frequency of the positive class
        """
        actual_outputs: pd.Series = self.sample.target

        return cast(int, (actual_outputs == self._positive_class()).sum()) / len(
            actual_outputs
        )

    def _positive_class(self) -> Any:
        """
        The label of the positive class of the binary classifier being simulated.
        """
        classifier = self.model

        try:
            return classifier.classes_[-1]

        except AttributeError:
            log.warning(
                f"{type(classifier).__name__} does not define classes_ attribute"
            )
            return "positive class"

    @staticmethod
    def _expected_learner_type() -> Type[ClassifierDF]:
        return ClassifierDF

    @staticmethod
    def _simulate(
        model: ClassifierDF, x: pd.DataFrame, name: str, value: Any
    ) -> Tuple[float, float]:
        probabilities: pd.DataFrame = model.predict_proba(
            BaseUnivariateSimulator._set_constant_feature_value(x, name, value)
        )
        if probabilities.shape[1] != 2:
            raise TypeError("only binary classifiers are supported")
        return BaseUnivariateSimulator._aggregate_simulation_results(
            probabilities.iloc[:, 1]
        )


@inheritdoc(match="[see superclass]")
class UnivariateTargetSimulator(UnivariateRegressionSimulator):
    """
    Univariate simulation of the absolute output of a regression model.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioner`
    object.

    For each value `v[j]` of the partitioning, a :class:`.Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then the regressor is used to predict the output for all
    observations, and the mean output across all observations is calculated
    for each regressor and value `v[j]`,
    along with the standard error of the mean as a basis of obtaining confidence
    intervals.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    # defined in superclass, repeated here for Sphinx
    n_jobs: Optional[int]

    # defined in superclass, repeated here for Sphinx
    shared_memory: Optional[bool]

    # defined in superclass, repeated here for Sphinx
    pre_dispatch: Optional[Union[str, int]]

    # defined in superclass, repeated here for Sphinx
    verbose: Optional[int]

    # defined in superclass, repeated here for Sphinx
    model: RegressorDF

    # defined in superclass, repeated here for Sphinx
    sample: Sample

    # defined in superclass, repeated here for Sphinx
    confidence_level: float

    @property
    def output_unit(self) -> str:
        """[see superclass]"""
        return f"Mean predicted target ({self.sample.target_name})"


@inheritdoc(match="[see superclass]")
class UnivariateUpliftSimulator(UnivariateRegressionSimulator):
    """
    Univariate simulation of the relative uplift of the output of a regression model.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioner`
    object.

    For each value `v[j]` of the partitioning, a :class:`.Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then the regressor is used to predict the output for all
    observations, and the mean output across all observations is calculated
    for each regressor and value `v[j]`,
    along with the standard error of the mean as a basis of obtaining confidence
    intervals.
    The simulation result is determined as the mean *uplift*, i.e., the mean
    predicted difference of the historical expectation value of the target,
    for each `v[j]`.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    # defined in superclass, repeated here for Sphinx
    n_jobs: Optional[int]

    # defined in superclass, repeated here for Sphinx
    shared_memory: Optional[bool]

    # defined in superclass, repeated here for Sphinx
    pre_dispatch: Optional[Union[str, int]]

    # defined in superclass, repeated here for Sphinx
    verbose: Optional[int]

    # defined in superclass, repeated here for Sphinx
    model: RegressorDF

    # defined in superclass, repeated here for Sphinx
    sample: Sample

    # defined in superclass, repeated here for Sphinx
    confidence_level: float

    @property
    def output_unit(self) -> str:
        """[see superclass]"""
        return f"Mean predicted uplift ({self.sample.target_name})"

    def baseline(self) -> float:
        """
        The baseline of uplift simulations is always ``0.0``

        :return: 0.0
        """
        return 0.0

    def simulate_feature(
        self,
        feature_name: str,
        *,
        partitioner: Partitioner[T_Values],
        **partitioner_params: Any,
    ) -> UnivariateSimulationResult[T_Values]:
        """[see superclass]"""

        result = super().simulate_feature(
            feature_name=feature_name, partitioner=partitioner, **partitioner_params
        )

        # offset the mean values to get uplift instead of absolute outputs
        result.data.loc[
            :,
            [
                UnivariateSimulationResult.COL_MEAN,
                UnivariateSimulationResult.COL_LOWER_BOUND,
                UnivariateSimulationResult.COL_UPPER_BOUND,
            ],
        ] -= self.expected_output()

        return result


__tracker.validate()
