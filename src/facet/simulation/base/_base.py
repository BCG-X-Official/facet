"""
Core implementation of :mod:`facet.simulation.base`
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import (
    Any,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from pytools.api import AllTracker
from pytools.parallelization import Job, JobRunner, ParallelizableMixin
from sklearndf import RegressorDF, SupervisedLearnerDF

from facet.data import Sample
from facet.data.partition import Partitioner
from facet.simulation._result import UnivariateSimulationResult

log = logging.getLogger(__name__)

__all__ = [
    "BaseUnivariateSimulator",
    "UnivariateRegressionSimulator",
]


#
# Type variables
#

T_Value = TypeVar("T_Value", bound=np.generic)
T_SupervisedLearnerDF = TypeVar("T_SupervisedLearnerDF", bound=SupervisedLearnerDF)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


class BaseUnivariateSimulator(
    ParallelizableMixin, Generic[T_SupervisedLearnerDF], metaclass=ABCMeta
):
    """
    Base class for univariate simulations.
    """

    # defined in superclass, repeated here for Sphinx
    n_jobs: Optional[int]

    # defined in superclass, repeated here for Sphinx
    shared_memory: Optional[bool]

    # defined in superclass, repeated here for Sphinx
    pre_dispatch: Optional[Union[str, int]]

    # defined in superclass, repeated here for Sphinx
    verbose: Optional[int]

    #: The learner pipeline used to conduct simulations
    model: T_SupervisedLearnerDF

    #: The sample to be used in baseline calculations and simulations
    sample: Sample

    #: The width of the confidence interval used to calculate the lower/upper bound
    #: of the simulation
    confidence_level: float

    def __init__(
        self,
        model: T_SupervisedLearnerDF,
        sample: Sample,
        *,
        confidence_level: float = 0.95,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param model: a fitted learner to use for calculating simulated outputs
        :param sample: the sample to be used for baseline calculations and simulations
        :param confidence_level: the width :math:`\\alpha` of the confidence interval
            to be estimated for simulation results
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if not isinstance(model, self._expected_learner_type()):
            raise TypeError(
                "arg model must be a learner of type "
                f"{self._expected_learner_type().__name__}."
            )

        if not model.is_fitted:
            raise ValueError("arg model must be fitted")

        if isinstance(sample.target_name, list):
            raise NotImplementedError("multi-output simulations are not supported")

        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"arg confidence_level={confidence_level} "
                "must range between 0.0 and 1.0 (exclusive)"
            )

        self.model = model
        self.sample = sample
        self.output_name = sample.target_name
        self.confidence_level = confidence_level

    # add parallelization parameters to __init__ docstring
    __init__.__doc__ = cast(str, __init__.__doc__) + cast(
        str, ParallelizableMixin.__init__.__doc__
    )

    def simulate_feature(
        self,
        feature_name: str,
        *,
        partitioner: Partitioner[T_Value],
        **partitioner_params: Any,
    ) -> UnivariateSimulationResult[T_Value]:
        """
        Simulate the average target uplift when fixing the value of the given feature
        across all observations.

        Simulations are run for a set of values determined by the given partitioner,
        which is fitted to the observed values for the feature being simulated.

        :param feature_name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for
        :param partitioner_params: additional parameters to pass to the partitioner
        :return: a mapping of output names to simulation results
        """

        sample = self.sample

        mean, sem = self._simulate_feature_with_values(
            feature_name=feature_name,
            simulation_values=partitioner.fit(
                sample.features.loc[:, feature_name], **partitioner_params
            ).partitions_,
        )
        return UnivariateSimulationResult(
            partitioner=partitioner,
            mean=mean,
            sem=sem,
            feature_name=feature_name,
            output_name=self.output_name,
            output_unit=self.output_unit,
            baseline=self.baseline(),
            confidence_level=self.confidence_level,
        )

    @property
    @abstractmethod
    def output_unit(self) -> str:
        """
        Unit of the output values calculated by the simulation.
        """

    def baseline(self) -> float:
        """
        Calculate the expectation value of the simulation result, based on historically
        observed actuals.

        :return: the expectation value of the simulation results
        """
        return self.expected_output()

    @abstractmethod
    def expected_output(self) -> float:
        """
        Calculate the expectation value of the actual model output, based on
        historically observed actuals.

        :return: the expectation value of the actual model output
        """
        pass

    @staticmethod
    @abstractmethod
    def _expected_learner_type() -> Type[T_SupervisedLearnerDF]:
        pass

    @staticmethod
    @abstractmethod
    def _simulate(
        model: T_SupervisedLearnerDF, x: pd.DataFrame, name: str, value: Any
    ) -> Tuple[float, float]:
        pass

    @staticmethod
    def _set_constant_feature_value(
        x: pd.DataFrame, feature_name: str, value: Any
    ) -> pd.DataFrame:
        return x.assign(
            **{
                feature_name: np.full(
                    shape=len(x),
                    fill_value=value,
                    dtype=x.loc[:, feature_name].dtype,
                )
            }
        )

    @staticmethod
    def _aggregate_simulation_results(predictions: pd.Series) -> Tuple[float, float]:
        # generate summary stats for a series of predictions
        return predictions.mean(), predictions.sem()

    def _simulate_feature_with_values(
        self,
        feature_name: str,
        simulation_values: Sequence[T_Value],
    ) -> Tuple[Sequence[float], Sequence[float]]:
        """
        Run a simulation on a feature.

        For each simulation value, compute the mean and sem of predictions when
        substituting the value for the feature being simulated.

        :param feature_name: name of the feature to use in the simulation
        :param simulation_values: values to use in the simulation
        :return: a tuple with mean predictions and standard errors of mean predictions
            for each partition
        """

        if feature_name not in self.sample.features.columns:
            raise ValueError(f"feature not in sample: {feature_name}")

        # for a list of values to be simulated, calculate a sequence of mean predictions
        # and a sequence of standard errors of those means
        features = self.sample.features

        outputs_mean_sem: Iterable[Tuple[float, float]] = JobRunner.from_parallelizable(
            self
        ).run_jobs(
            Job.delayed(self._simulate)(self.model, features, feature_name, value)
            for value in simulation_values
        )

        outputs_mean, outputs_sem = zip(*outputs_mean_sem)
        return outputs_mean, outputs_sem


class UnivariateRegressionSimulator(
    BaseUnivariateSimulator[RegressorDF], metaclass=ABCMeta
):
    """
    Base class for univariate simulations using regression models.
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

    def expected_output(self) -> float:
        """
        Calculate the mean of actually observed values for the target.

        :return: mean observed value of the target
        """
        return cast(float, self.sample.target.mean())

    @staticmethod
    def _expected_learner_type() -> Type[RegressorDF]:
        return RegressorDF

    @staticmethod
    def _simulate(
        model: RegressorDF, x: pd.DataFrame, name: str, value: Any
    ) -> Tuple[float, float]:
        predictions = model.predict(
            X=BaseUnivariateSimulator._set_constant_feature_value(x, name, value)
        )
        assert predictions.ndim == 1, "single-target regressor required"
        return BaseUnivariateSimulator._aggregate_simulation_results(predictions)
