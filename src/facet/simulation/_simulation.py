"""
Core implementation of :mod:`facet.simulation`
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
from scipy import stats

from pytools.api import AllTracker, inheritdoc
from pytools.parallelization import Job, JobRunner, ParallelizableMixin
from sklearndf import ClassifierDF, LearnerDF, RegressorDF

from ..data import Sample
from ..data.partition import Partitioner

log = logging.getLogger(__name__)

__all__ = [
    "UnivariateSimulationResult",
    "BaseUnivariateSimulator",
    "UnivariateProbabilitySimulator",
    "UnivariateTargetSimulator",
    "UnivariateUpliftSimulator",
]


#
# Type variables
#

T_LearnerDF = TypeVar("T_LearnerDF", bound=LearnerDF)
T_Partition = TypeVar("T_Partition")


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class UnivariateSimulationResult(Generic[T_Partition]):
    """
    Summary result of a univariate simulation.
    """

    #: The simulation result as a data frame, indexed by the central values of the
    #: partitions for which the simulation was run, with the following columns:
    #:
    #: - :attr:`.COL_MEAN`: the mean predictions for the simulated values
    #: - :attr:`.COL_SEM`: the standard errors of the mean predictions
    #: - :attr:`.COL_LOWER_BOUND`: the lower bounds of the confidence intervals for the
    #:   simulation outcomes, based on mean, standard error of the mean, and
    #:   :attr:`confidence_level`
    #: - :attr:`.COL_UPPER_BOUND`: the upper bounds of the confidence intervals for the
    #:   simulation outcomes, based on mean, standard error of the mean, and
    #:   :attr:`confidence_level`
    data: pd.DataFrame

    #: The partitioner used to generate feature values to be simulated.
    partitioner: Partitioner

    #: Name of the simulated feature.
    feature_name: str

    #: Name of the target for which outputs are simulated.
    output_name: str

    #: The unit of the simulated outputs (e.g., uplift or class probability).
    output_unit: str

    #: The average observed actual output, acting as the baseline of the simulation.
    baseline: float

    #: The width :math:`\alpha` of the confidence interval
    #: determined by bootstrapping, with :math:`0 < \alpha < 1`.
    confidence_level: float

    #: The name of the column index of attribute :attr:`.output`, denoting partitions
    #: represented by their central values or by a category.
    IDX_PARTITION = "partition"

    #: The name of a series of mean simulated values per partition.
    COL_MEAN = "mean"

    #: The name of a series of standard errors of mean simulated values per partition.
    COL_SEM = "sem"

    #: The name of a series of lower CI bounds of simulated values per partition.
    COL_LOWER_BOUND = "lower_bound"

    #: The name of a series of upper CI bounds of simulated values per partition.
    COL_UPPER_BOUND = "upper_bound"

    def __init__(
        self,
        *,
        partitioner: Partitioner,
        mean: Sequence[float],
        sem: Sequence[float],
        feature_name: str,
        output_name: str,
        output_unit: str,
        baseline: float,
        confidence_level: float,
    ) -> None:
        """
        :param partitioner: the partitioner used to generate feature values to be
            simulated
        :param mean: mean predictions for the values representing each partition
        :param sem: standard errors of the mean predictions for the values representing
            each partition
        :param feature_name: name of the simulated feature
        :param output_name: name of the target for which outputs are simulated
        :param output_unit: the unit of the simulated outputs
            (e.g., uplift or class probability)
        :param baseline: the average observed actual output, acting as the baseline
            of the simulation
        :param confidence_level: the width of the confidence interval determined by
            bootstrapping, ranging between 0.0 and 1.0 (exclusive)
        """
        super().__init__()

        if not partitioner.is_fitted:
            raise ValueError("arg partitioner must be fitted")

        n_partitions = len(partitioner.partitions_)

        for seq, seq_name in [(mean, "mean"), (sem, "sem")]:
            if len(seq) != n_partitions:
                raise ValueError(
                    f"length of arg {seq_name} must correspond to "
                    f"the number of partitions (n={n_partitions})"
                )

        if not (0.0 < confidence_level < 1.0):
            raise ValueError(
                f"arg confidence_level={confidence_level} is not "
                "in the range between 0.0 and 1.0 (exclusive)"
            )

        self.partitioner = partitioner
        self.feature_name = feature_name
        self.output_name = output_name
        self.output_unit = output_unit
        self.baseline = baseline
        self.confidence_level = confidence_level

        # convert mean and sem to numpy arrays
        mean_arr = np.array(mean)
        sem_arr = np.array(sem)

        # get the width of the confidence interval (this is a negative number)
        ci_width = stats.norm.ppf((1.0 - self.confidence_level) / 2.0) * sem_arr

        self.data = pd.DataFrame(
            data={
                UnivariateSimulationResult.COL_MEAN: mean_arr,
                UnivariateSimulationResult.COL_SEM: sem_arr,
                UnivariateSimulationResult.COL_LOWER_BOUND: mean_arr + ci_width,
                UnivariateSimulationResult.COL_UPPER_BOUND: mean_arr - ci_width,
            },
            index=pd.Index(
                partitioner.partitions_, name=UnivariateSimulationResult.IDX_PARTITION
            ),
        )


class BaseUnivariateSimulator(
    ParallelizableMixin, Generic[T_LearnerDF], metaclass=ABCMeta
):
    """
    Base class for univariate simulations.
    """

    #: The learner pipeline used to conduct simulations
    model: T_LearnerDF

    #: The sample to be used in baseline calculations and simulations
    sample: Sample

    #: The width of the confidence interval used to calculate the lower/upper bound
    #: of the simulation
    confidence_level: float

    def __init__(
        self,
        model: T_LearnerDF,
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
        self.output_name = cast(str, sample.target_name)
        self.confidence_level = confidence_level

    # add parallelization parameters to __init__ docstring
    __init__.__doc__ = cast(str, __init__.__doc__) + cast(
        str, ParallelizableMixin.__init__.__doc__
    )

    def simulate_feature(
        self,
        feature_name: str,
        *,
        partitioner: Partitioner[T_Partition],
    ) -> UnivariateSimulationResult:
        """
        Simulate the average target uplift when fixing the value of the given feature
        across all observations.

        :param feature_name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for
        :return: a mapping of output names to simulation results
        """

        sample = self.sample

        mean, sem = self._simulate_feature_with_values(
            feature_name=feature_name,
            simulation_values=partitioner.fit(
                sample.features.loc[:, feature_name]
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
    def _expected_learner_type() -> Type[T_LearnerDF]:
        pass

    @staticmethod
    @abstractmethod
    def _simulate(
        model: T_LearnerDF, x: pd.DataFrame, name: str, value: Any
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
        simulation_values: Sequence[T_Partition],
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
        actual_outputs = self.sample.target

        return (actual_outputs == self._positive_class()).sum() / len(actual_outputs)

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


class _UnivariateRegressionSimulator(
    BaseUnivariateSimulator[RegressorDF], metaclass=ABCMeta
):
    def expected_output(self) -> float:
        """
        Calculate the mean of actually observed values for the target.

        :return: mean observed value of the target
        """
        return self.sample.target.mean()

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


@inheritdoc(match="[see superclass]")
class UnivariateTargetSimulator(_UnivariateRegressionSimulator):
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

    @property
    def output_unit(self) -> str:
        """[see superclass]"""
        return f"Mean predicted target ({self.sample.target_name})"


@inheritdoc(match="[see superclass]")
class UnivariateUpliftSimulator(_UnivariateRegressionSimulator):
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
        self, feature_name: str, *, partitioner: Partitioner[T_Partition]
    ) -> UnivariateSimulationResult:
        """[see superclass]"""

        result = super().simulate_feature(
            feature_name=feature_name, partitioner=partitioner
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
