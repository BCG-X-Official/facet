"""
Core implementation of :mod:`facet.simulation`
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.parallelization import ParallelizableMixin
from sklearndf import LearnerDF
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from ..crossfit import LearnerCrossfit
from ..data import Sample
from .partition import Partitioner

log = logging.getLogger(__name__)

__all__ = [
    "UnivariateSimulation",
    "BaseUnivariateSimulator",
    "UnivariateProbabilitySimulator",
    "UnivariateUpliftSimulator",
]

#
# Constants
#

# if True, use the full available sample to carry out simulations; otherwise only
# use the train sample of each fold
_SIMULATE_FULL_SAMPLE = True


#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)
T_Number = TypeVar("T_Number", int, float)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class UnivariateSimulation(NamedTuple, Generic[T_Number]):
    """
    Summary result of a univariate simulation.
    """

    #: name of the simulated feature
    feature: str

    #: name of the simulated target
    target: str

    #: the partitioner used to run the simulation
    partitioner: Partitioner

    #: the unit of the simulated outcomes (e.g., uplift or class probability)
    values_label: str

    #: the median of the distribution of simulation outcomes, for every partition
    values_median: Sequence[T_Number]

    #: the lower boundary of the confidence interval of simulated outcomes, for every
    #: partition
    values_lower: Sequence[T_Number]

    #: the upper boundary of the confidence interval of simulated outcomes, for every
    #: partition
    values_upper: Sequence[T_Number]

    #: the mean of the actual observed outputs, acting as the baseline of the
    values_baseline: T_Number

    #: the percentile used to determine the lower confidence interval from the
    #: distribution of simulated outcomes
    percentile_lower: float

    #: the percentile used to determine the upper confidence interval from the
    #: distribution of simulated outcomes
    percentile_upper: float


class BaseUnivariateSimulator(
    ParallelizableMixin, Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for univariate simulations.
    """

    COL_SPLIT_ID = "split_id"
    COL_VALUE = "value"

    def __init__(
        self,
        crossfit: LearnerCrossfit[T_LearnerPipelineDF],
        *,
        percentile_lower: float = 2.5,
        percentile_upper: float = 97.5,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ):
        """
        :param crossfit: cross-validated crossfit of a model for all observations \
        in a given sample
        :param percentile_lower: lower bound of the confidence interval (default: 2.5)
        :param percentile_upper: upper bound of the confidence interval (default: 97.5)
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if not isinstance(crossfit.pipeline, self._expected_pipeline_type()):
            raise TypeError(
                f"arg crossfit must fit a pipeline of type "
                f"{self._expected_pipeline_type().__name__}."
            )

        if not crossfit.is_fitted:
            raise ValueError("arg crossfit expected to be fitted")

        if isinstance(crossfit.sample_.target_name, list):
            raise NotImplementedError("multi-output simulations are not supported")

        if not 0 <= percentile_lower <= 100:
            raise ValueError(
                f"arg percentile_lower={percentile_lower} must be in the range"
                "from 0 to 100"
            )
        if not 0 <= percentile_upper <= 100:
            raise ValueError(
                f"arg percentile_upper={percentile_upper} must be in the range"
                "from 0 to 1"
            )

        self.crossfit = crossfit
        self.percentile_upper = percentile_upper
        self.percentile_lower = percentile_lower

    # add parallelization parameters to __init__ docstring
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    def simulate_feature(
        self, name: str, partitioner: Partitioner
    ) -> UnivariateSimulation:
        """
        Simulate the average target uplift when fixing the value of the given feature
        across all observations.

        :param name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for

        :return a mapping of output names to simulation results
        """

        sample = self.crossfit.sample_

        if isinstance(sample.target_name, list):
            raise NotImplementedError("multi-output simulations are not supported")

        simulation_values = partitioner.fit(sample.features.loc[:, name]).partitions_
        simulation_results = self._aggregate_simulation_results(
            results_per_split=self._simulate_feature_with_values(
                feature_name=name, simulation_values=simulation_values
            )
        )
        return UnivariateSimulation(
            feature=name,
            target=sample.target_name,
            partitioner=partitioner,
            values_label=self.values_label,
            values_median=simulation_results.iloc[:, 1].values,
            values_lower=simulation_results.iloc[:, 0].values,
            values_upper=simulation_results.iloc[:, 2].values,
            values_baseline=self.baseline,
            percentile_lower=self.percentile_lower,
            percentile_upper=self.percentile_upper,
        )

    def simulate_actuals(self) -> pd.Series:
        """
        Run a simulation by predicting the outcome based on the actual feature values
        across all splits of the crossfit.

        In the case of regressors, for each split determine the relative deviation of
        the mean predicted target from the mean actual target.
        For any of the splits, 0 indicates no deviation, and, for example, 0.01
        indicates that the mean predicted target is 1% higher than the mean actual
        targets of a given split.

        In the case of binary classifiers, for each split determine the mean predicted
        probability of the positive class.

        The breadth and offset of this actual simulation is an indication of how the
        bias of the model underlying the simulation contributes to the uncertainty of
        simulations produced with method :meth:`.simulate_features`.

        :return: series mapping split IDs to simulation results based on actual \
            feature values
        """

        sample = self.crossfit.sample_

        with self._parallel() as parallel:
            result: List[float] = parallel(
                self._delayed(self._simulate_actuals)(
                    model=model,
                    subsample=(
                        sample
                        if _SIMULATE_FULL_SAMPLE
                        else sample.subsample(iloc=test_indices)
                    ),
                )
                for (model, (_, test_indices)) in zip(
                    self.crossfit.models(), self.crossfit.splits()
                )
            )

        return pd.Series(
            index=pd.RangeIndex(len(result), name=BaseUnivariateSimulator.COL_SPLIT_ID),
            data=result,
            name=BaseUnivariateSimulator.COL_VALUE,
        )

    @property
    @abstractmethod
    def values_label(self) -> str:
        """
        Designation for the values calculated in the simulation
        """

    @property
    @abstractmethod
    def baseline(self) -> float:
        """
        The mean of actual observed outputs
        """

    @staticmethod
    @abstractmethod
    def _expected_pipeline_type() -> Type[T_LearnerPipelineDF]:
        pass

    @staticmethod
    @abstractmethod
    def _simulate(
        model: T_LearnerPipelineDF, x: pd.DataFrame, actual_outcomes: pd.Series
    ) -> float:
        pass

    @staticmethod
    @abstractmethod
    def _simulate_actuals(model: T_LearnerPipelineDF, subsample: Sample) -> float:
        pass

    def _simulate_feature_with_values(
        self, feature_name: str, simulation_values: Sequence[Any]
    ) -> pd.Series:
        """
        Run a simulation on a feature.

        For each combination of crossfit and feature value, compute the simulation
        result when substituting a given fixed value for the feature being simulated.

        :param feature_name: name of the feature to use in the simulation
        :param simulation_values: values to use in the simulation
        :return: data frame with three columns: ``crossfit_id``, ``parameter_value`` and
          ``simulation_result``.
        """

        sample = self.crossfit.sample_

        if feature_name not in sample.features.columns:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        with self._parallel() as parallel:
            simulation_results: List[List[Union[float]]] = parallel(
                self._delayed(UnivariateUpliftSimulator._simulate_values_for_split)(
                    model=model,
                    subsample=(
                        sample
                        if _SIMULATE_FULL_SAMPLE
                        else sample.subsample(iloc=test_indices)
                    ),
                    feature_name=feature_name,
                    simulated_values=simulation_values,
                    simulate_fn=self._simulate,
                )
                for (model, (_, test_indices)) in zip(
                    self.crossfit.models(), self.crossfit.splits()
                )
            )

        return pd.concat(
            pd.Series(index=simulation_values, data=result)
            for _, result in enumerate(simulation_results)
        )

    @staticmethod
    def _simulate_values_for_split(
        model: LearnerDF,
        subsample: Sample,
        feature_name: str,
        simulated_values: Optional[Sequence[Any]],
        simulate_fn: Callable[[LearnerDF, pd.DataFrame, pd.Series], float],
    ) -> List[float]:
        # for a list of values to be simulated, return a list of absolute target changes

        n_observations = len(subsample)
        features = subsample.features
        feature_dtype = features.loc[:, feature_name].dtype

        actual_outcomes = subsample.target

        return [
            simulate_fn(
                model,
                features.assign(
                    **{
                        feature_name: np.full(
                            shape=n_observations, fill_value=value, dtype=feature_dtype
                        )
                    }
                ),
                actual_outcomes,
            )
            for value in simulated_values
        ]

    def _aggregate_simulation_results(
        self, results_per_split: pd.Series
    ) -> pd.DataFrame:
        """
        Aggregate uplift values computed by ``simulate_feature``.

        For each parameter value, the percentile of uplift values (in the
        ``relative_yield_change`` column) are computed.

        :param results_per_split: data frame with columns
            ``crossfit_id``, ``parameter_value``, and ``relative_yield_change``
        :return: data frame with 3 columns ``percentile_<min>``, ``percentile_50``,
          ``percentile_<max>`` where min/max are the min and max percentiles
        """

        def percentile(n: int) -> Callable[[float], float]:
            """
            Return the function computed the n-th percentile.

            :param n: the percentile to compute; int between 0 and 100
            :return: the n-th percentile function
            """

            def percentile_(x: float):
                """n-th percentile function"""
                return np.percentile(x, n)

            percentile_.__name__ = f"percentile_{n}"

            return percentile_

        return results_per_split.groupby(level=0, observed=True, sort=False).agg(
            [percentile(p) for p in (self.percentile_lower, 50, self.percentile_upper)]
        )


@inheritdoc(match="[see superclass]")
class UnivariateProbabilitySimulator(BaseUnivariateSimulator[ClassifierPipelineDF]):
    """
    Univariate simulation for predicted probability based on a binary classifier.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioning`
    object.

    For each value `v[j]` of the partitioning, a :class:`Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then all classifiers of a :class:`LearnerCrossfit` are used in turn to each predict
    the probability of the positive class for all observations, and the mean probability
    across all observations is calculated for each classifier, resulting in a
    distribution of mean predicted probabilities for each value `v[j]`.

    For each `v[j]`, the median and the lower and upper confidence bounds are retained.

    Hence the result of the simulation is a series of `n` medians, lower and upper
    confidence  bounds; one each for every value in the range of simulated values.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    @property
    def values_label(self) -> str:
        """[see superclass]"""
        return f"probability({self._positive_class()})"

    @property
    def baseline(self) -> float:
        """
        Calculate the actual observed frequency of the positive class as the baseline
        of the simulation
        :return: observed frequency of the positive class
        """
        actual_target: pd.Series = self.crossfit.sample_.target
        assert isinstance(actual_target, pd.Series), "sample has one single target"

        return actual_target.loc[actual_target == self._positive_class()].sum() / len(
            actual_target
        )

    def _positive_class(self) -> Any:
        """
        The label of the positive class of the binary classifier being simulated
        """
        classifier = self.crossfit.pipeline.final_estimator

        try:
            return classifier.classes_[-1]

        except AttributeError:
            log.warning(
                f"{type(classifier).__name__} does not define classes_ attribute"
            )
            return "positive class"

    @staticmethod
    def _expected_pipeline_type() -> Type[ClassifierPipelineDF]:
        return ClassifierPipelineDF

    @staticmethod
    def _simulate(
        model: ClassifierPipelineDF, x: pd.DataFrame, actual_outcomes: pd.Series
    ) -> float:
        probabilities: pd.DataFrame = model.predict_proba(x)
        if probabilities.shape[1] != 2:
            raise TypeError("only binary classifiers are supported")
        return probabilities.iloc[:, 1].mean()

    @staticmethod
    def _simulate_actuals(model: ClassifierPipelineDF, subsample: Sample) -> float:
        # return relative difference between actual and predicted target
        probabilities = model.predict_proba(X=subsample.features)

        if probabilities.shape[1] != 2:
            raise TypeError("only binary classifiers are supported")

        return probabilities.iloc[:, 1].mean(axis=0)


@inheritdoc(match="[see superclass]")
class _UnivariateTargetSimulator(BaseUnivariateSimulator[RegressorPipelineDF]):
    """
    Univariate simulation for absolute target values based on a regression model.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioning`
    object.

    For each value `v[j]` of the partitioning, a :class:`Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then all regressors of a :class:`LearnerCrossfit` are used in turn to each predict
    the target for all observations, and the mean of the predicted targets is calculated
    for each regressor and value `v[j]`. The outcome is a distribution of mean predicted
    targets for each `v[j]`.

    For each `v[j]`, the median and the lower and upper confidence bounds of that
    distribution are retained.

    Hence the result of the simulation is a series of `n` medians, lower and upper
    confidence bounds; one each for every value in the range of simulated values.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    @property
    def values_label(self) -> str:
        """[see superclass]"""
        return f"Mean predicted target ({self.crossfit.sample_.target_name})"

    @property
    def baseline(self) -> float:
        """
        The baseline of uplift simulations is always ``0.0``
        """
        return 0.0

    @staticmethod
    def _expected_pipeline_type() -> Type[RegressorPipelineDF]:
        return RegressorPipelineDF

    @staticmethod
    def _simulate(
        model: RegressorPipelineDF, x: pd.DataFrame, actual_outcomes: pd.Series
    ) -> float:
        return model.predict(x).mean(axis=0) - actual_outcomes.mean(axis=0)

    @staticmethod
    def _simulate_actuals(model: RegressorPipelineDF, subsample: Sample) -> float:
        # return relative difference between actual and predicted target
        return (
            model.predict(X=subsample.features).mean(axis=0)
            / subsample.target.mean(axis=0)
            - 1.0
        )


@inheritdoc(match="[see superclass]")
class UnivariateUpliftSimulator(BaseUnivariateSimulator[RegressorPipelineDF]):
    """
    Univariate simulation for target uplift based on a regression model.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioning`
    object.

    For each value `v[j]` of the partitioning, a :class:`Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then all regressors of a :class:`LearnerCrossfit` are used in turn to each predict
    the target for all observations, and the mean difference of the predicted targets
    from the actual (known) targets across all observations is calculated for each
    regressor, resulting in a distribution of mean `uplift` amounts for each value
    `v[j]`.

    For each `v[j]`, the median uplift and the lower and upper confidence bounds are
    retained.

    Hence the result of the simulation is a series of `n` medians, lower and upper
    confidence bounds; one each for every value in the range of simulated values.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    @property
    def values_label(self) -> str:
        """[see superclass]"""
        return f"Mean predicted uplift ({self.crossfit.sample_.target_name})"

    @property
    def baseline(self) -> float:
        """
        The baseline of uplift simulations is always ``0.0``
        """
        return 0.0

    @staticmethod
    def _expected_pipeline_type() -> Type[RegressorPipelineDF]:
        return RegressorPipelineDF

    @staticmethod
    def _simulate(
        model: RegressorPipelineDF, x: pd.DataFrame, actual_outcomes: pd.Series
    ) -> float:
        return model.predict(x).mean(axis=0) - actual_outcomes.mean(axis=0)

    @staticmethod
    def _simulate_actuals(model: RegressorPipelineDF, subsample: Sample) -> float:
        # return relative difference between actual and predicted target
        return (
            model.predict(X=subsample.features).mean(axis=0)
            / subsample.target.mean(axis=0)
            - 1.0
        )


__tracker.validate()
