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
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.parallelization import Job, JobRunner, ParallelizableMixin
from sklearndf import LearnerDF
from sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)

from ..crossfit import LearnerCrossfit
from ..data import Sample
from ..data.partition import Partitioner
from ..validation import BaseBootstrapCV

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

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)
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

    #: The partitioner used to generate feature values to be simulated.
    partitioner: Partitioner

    #: The matrix of simulated outcomes, with columns representing partitions
    #: and rows representing bootstrap splits used to fit variations of the model.
    outputs: pd.DataFrame

    #: The name of a series of median simulated values per partition.
    COL_MEDIAN = "median"

    #: The name of a series of lower CI bounds of simulated values per partition.
    COL_LOWER_BOUND = "lower_bound"

    #: The name of a series of upper CI bounds of simulated values per partition.
    COL_UPPER_BOUND = "upper_bound"

    def __init__(
        self,
        *,
        feature_name: str,
        output_name: str,
        output_unit: str,
        baseline: float,
        confidence_level: float,
        partitioner: Partitioner,
        outputs: pd.DataFrame,
    ) -> None:
        """
        :param feature_name: name of the simulated feature
        :param output_name: name of the target for which outputs are simulated
        :param output_unit: the unit of the simulated outputs
            (e.g., uplift or class probability)
        :param baseline: the average observed actual output, acting as the baseline
            of the simulation
        :param confidence_level: the width of the confidence interval determined by
            bootstrapping, ranging between 0.0 and 1.0 (exclusive)
        :param partitioner: the partitioner used to generate feature values to be
            simulated
        :param outputs: matrix of simulated outcomes, with columns representing
            partitions and rows representing bootstrap splits used to fit variations
            of the model
        """
        super().__init__()

        assert (
            outputs.index.name == BaseUnivariateSimulator.IDX_SPLIT
        ), f"row index of arg outputs is named {BaseUnivariateSimulator.IDX_SPLIT}"
        assert outputs.columns.name == BaseUnivariateSimulator.IDX_PARTITION, (
            "column index of arg outputs is named "
            f"{BaseUnivariateSimulator.IDX_PARTITION}"
        )
        assert (
            0.0 < confidence_level < 1.0
        ), f"confidence_level={confidence_level} ranges between 0.0 and 1.0 (exclusive)"

        self.feature_name = feature_name
        self.output_name = output_name
        self.output_unit = output_unit
        self.baseline = baseline
        self.confidence_level = confidence_level
        self.partitioner = partitioner
        self.outputs = outputs

    def outputs_median(self) -> pd.Series:
        """
        Calculate the medians of the distribution of simulation outcomes,
        for every partition.

        :return: a series of medians, indexed by the central values of the partitions
            for which the simulation was run
        """
        return self.outputs.median().rename(UnivariateSimulationResult.COL_MEDIAN)

    def outputs_lower_bound(self) -> pd.Series:
        """
        Calculate the lower CI bounds of the distribution of simulation outcomes,
        for every partition.

        :return: a series of medians, indexed by the central values of the partitions
            for which the simulation was run
        """
        return self.outputs.quantile(q=(1.0 - self.confidence_level) / 2.0).rename(
            UnivariateSimulationResult.COL_LOWER_BOUND
        )

    def outputs_upper_bound(self) -> pd.Series:
        """
        Calculate the lower CI bounds of the distribution of simulation outcomes,
        for every partition.

        :return: a series of medians, indexed by the central values of the partitions
            for which the simulation was run
        """
        return self.outputs.quantile(
            q=1.0 - (1.0 - self.confidence_level) / 2.0
        ).rename(UnivariateSimulationResult.COL_UPPER_BOUND)


class BaseUnivariateSimulator(
    ParallelizableMixin, Generic[T_LearnerPipelineDF], metaclass=ABCMeta
):
    """
    Base class for univariate simulations.
    """

    #: The name of the row index of attribute :attr:`.output`, denoting splits.
    IDX_SPLIT = "split"

    #: The name of the column index of attribute :attr:`.output`, denoting partitions
    #: represented by their central values or by a category.
    IDX_PARTITION = "partition"

    #: The name of a series of simulated outputs.
    COL_OUTPUT = "output"

    def __init__(
        self,
        crossfit: LearnerCrossfit[T_LearnerPipelineDF],
        *,
        confidence_level: float = 0.95,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param crossfit: cross-validated crossfit of a model for all observations
            in a given sample
        :param confidence_level: the width :math:`\\alpha` of the confidence interval
            determined by bootstrapping, with :math:`0 < \\alpha < 1`;
            for reliable CI estimates the number of splits in the crossfit should be
            at least :math:`n = \\frac{50}{1 - \\alpha}`, e.g. :math:`n = 1000` for
            :math:`\\alpha = 0.95`
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if not isinstance(crossfit.pipeline, self._expected_pipeline_type()):
            raise TypeError(
                "arg crossfit must fit a pipeline of type "
                f"{self._expected_pipeline_type().__name__}."
            )

        if not crossfit.is_fitted:
            raise ValueError("arg crossfit expected to be fitted")

        if isinstance(crossfit.sample_.target_name, list):
            raise NotImplementedError("multi-output simulations are not supported")

        if not 0.0 < confidence_level < 1.0:
            raise ValueError(
                f"arg confidence_level={confidence_level} "
                "must range between 0.0 and 1.0 (exclusive)"
            )

        if not isinstance(crossfit.cv, BaseBootstrapCV):
            log.warning(
                "arg crossfit.cv should be a bootstrap cross-validator "
                f"but is a {type(crossfit.cv).__name__}"
            )

        min_splits = int(50 / (1.0 - confidence_level))
        if len(crossfit) < min_splits:
            log.warning(
                f"at least {min_splits} bootstrap splits are recommended for "
                f"reliable results with arg confidence_level={confidence_level}, "
                f"but arg crossfit.cv has only {len(crossfit)} splits"
            )

        self.crossfit = crossfit
        self.confidence_level = confidence_level

    # add parallelization parameters to __init__ docstring
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    def simulate_feature(
        self, feature_name: str, *, partitioner: Partitioner[T_Partition]
    ) -> UnivariateSimulationResult:
        """
        Simulate the average target uplift when fixing the value of the given feature
        across all observations.

        :param feature_name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for
        :return: a mapping of output names to simulation results
        """

        sample = self.crossfit.sample_

        if isinstance(sample.target_name, list):
            raise NotImplementedError("multi-output simulations are not supported")

        return UnivariateSimulationResult(
            feature_name=feature_name,
            output_name=sample.target_name,
            output_unit=self.output_unit,
            baseline=self.baseline(),
            confidence_level=self.confidence_level,
            partitioner=partitioner,
            outputs=(
                self._simulate_feature_with_values(
                    feature_name=feature_name,
                    simulation_values=(
                        partitioner.fit(
                            sample.features.loc[:, feature_name]
                        ).partitions_
                    ),
                )
            ),
        )

    def simulate_actuals(self) -> pd.Series:
        r"""
        For each test split :math:`\mathrm{T}_i` in this simulator's
        crossfit, predict the outputs for all test samples given their actual
        feature values, and calculate the absolute deviation from the mean of all actual
        outputs of the entire sample
        :math:`\frac{1}{n}\sum_{j \in \mathrm{T}_i}\hat y_j - \bar y`.

        The spread and offset of these deviations can serve as an indication of how the
        bias of the model contributes to the uncertainty of simulations produced with
        method :meth:`.simulate_feature`.

        :return: series mapping split IDs to deviations of simulated mean outputs
        """

        sample = self.crossfit.sample_
        y_mean = self.expected_output()

        result: List[float] = JobRunner.from_parallelizable(self).run_jobs(
            *(
                Job.delayed(self._simulate_actuals)(
                    model, subsample.features, y_mean, self._simulate
                )
                for (model, (_, test_indices)) in zip(
                    self.crossfit.models(), self.crossfit.splits()
                )
                for subsample in (sample.subsample(iloc=test_indices),)
            )
        )

        return pd.Series(
            data=result, name=BaseUnivariateSimulator.COL_OUTPUT
        ).rename_axis(index=BaseUnivariateSimulator.IDX_SPLIT)

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
    def _expected_pipeline_type() -> Type[T_LearnerPipelineDF]:
        pass

    @staticmethod
    @abstractmethod
    def _simulate(model: T_LearnerPipelineDF, x: pd.DataFrame) -> float:
        pass

    def _simulate_feature_with_values(
        self, feature_name: str, simulation_values: Sequence[T_Partition]
    ) -> pd.DataFrame:
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

        simulation_results_per_split: List[np.ndarray] = JobRunner.from_parallelizable(
            self
        ).run_jobs(
            *(
                Job.delayed(UnivariateUpliftSimulator._simulate_values_for_split)(
                    model=model,
                    subsample=sample.subsample(iloc=test_indices),
                    feature_name=feature_name,
                    simulated_values=simulation_values,
                    simulate_fn=self._simulate,
                )
                for (model, (_, test_indices)) in zip(
                    self.crossfit.models(), self.crossfit.splits()
                )
            )
        )

        return pd.DataFrame(
            simulation_results_per_split, columns=simulation_values
        ).rename_axis(
            index=BaseUnivariateSimulator.IDX_SPLIT,
            columns=BaseUnivariateSimulator.IDX_PARTITION,
        )

    @staticmethod
    def _simulate_values_for_split(
        model: LearnerDF,
        subsample: Sample,
        feature_name: str,
        simulated_values: Optional[Sequence[Any]],
        simulate_fn: Callable[[LearnerDF, pd.DataFrame], float],
    ) -> np.ndarray:
        # for a list of values to be simulated, return a list of absolute target changes

        n_observations = len(subsample)
        features = subsample.features
        feature_dtype = features.loc[:, feature_name].dtype

        return np.array(
            [
                simulate_fn(
                    model,
                    features.assign(
                        **{
                            feature_name: np.full(
                                shape=n_observations,
                                fill_value=value,
                                dtype=feature_dtype,
                            )
                        }
                    ),
                )
                for value in simulated_values
            ]
        )

    @staticmethod
    def _simulate_actuals(
        model: LearnerDF,
        x: pd.DataFrame,
        y_mean: float,
        simulate_fn: Callable[[LearnerDF, pd.DataFrame], float],
    ):
        return simulate_fn(model, x) - y_mean


@inheritdoc(match="[see superclass]")
class UnivariateProbabilitySimulator(BaseUnivariateSimulator[ClassifierPipelineDF]):
    """
    Univariate simulation of positive class probabilities based on a binary classifier.

    The simulation is carried out for one specific feature `x[i]` of a model, and for a
    range of values `v[1]`, …, `v[n]` for `f`, determined by a :class:`.Partitioner`
    object.

    For each value `v[j]` of the partitioning, a :class:`.Sample` of historical
    observations is modified by assigning value `v[j]` for feature `x[i]` for all
    observations, i.e., assuming that feature `x[i]` has the constant value `v[j]`.

    Then all classifiers of a :class:`.LearnerCrossfit` are used in turn to each predict
    the positive class probabilities for all observations, and the mean probability
    across all observations is calculated for each classifier and value `v[j]`.
    The simulation result is a set of `n` distributions of mean predicted probabilities
    across all classifiers -- one distribution for each `v[j]`.

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
        actual_outputs: pd.Series = self.crossfit.sample_.target
        assert isinstance(actual_outputs, pd.Series), "sample has one single target"

        return (actual_outputs == self._positive_class()).sum() / len(actual_outputs)

    def _positive_class(self) -> Any:
        """
        The label of the positive class of the binary classifier being simulated.
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
    def _simulate(model: ClassifierPipelineDF, x: pd.DataFrame) -> float:
        probabilities: pd.DataFrame = model.predict_proba(x)
        if probabilities.shape[1] != 2:
            raise TypeError("only binary classifiers are supported")
        return probabilities.iloc[:, 1].mean()


class _UnivariateRegressionSimulator(
    BaseUnivariateSimulator[RegressorPipelineDF], metaclass=ABCMeta
):
    def expected_output(self) -> float:
        """
        Calculate the mean of actually observed values for the target.

        :return: mean observed value of the target
        """
        actual_outputs: pd.Series = self.crossfit.sample_.target
        assert isinstance(actual_outputs, pd.Series), "sample has one single target"

        return actual_outputs.mean()

    @staticmethod
    def _expected_pipeline_type() -> Type[RegressorPipelineDF]:
        return RegressorPipelineDF

    @staticmethod
    def _simulate(model: RegressorPipelineDF, x: pd.DataFrame) -> float:
        return model.predict(X=x).mean(axis=0)


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

    Then all regressors of a :class:`.LearnerCrossfit` are used in turn to each predict
    the output for all observations, and the mean of the predicted outputs is calculated
    for each regressor and value `v[j]`. The simulation result is a set of `n`
    distributions of mean predicted targets across regressors -- one distribution for
    each `v[j]`.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    @property
    def output_unit(self) -> str:
        """[see superclass]"""
        return f"Mean predicted target ({self.crossfit.sample_.target_name})"


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

    Then all regressors of a :class:`.LearnerCrossfit` are used in turn to each predict
    the output for all observations, and the mean of the predicted outputs is calculated
    for each regressor and value `v[j]`. The simulation result is a set of `n`
    distributions of mean predicted target uplifts across regressors, i.e. the mean
    predicted difference of the historical expectation value of the target --
    one distribution for each `v[j]`.

    Note that sample weights are not taken into account for simulations; each
    observation has the same weight in the simulation even if different weights
    have been specified for the sample.
    """

    @property
    def output_unit(self) -> str:
        """[see superclass]"""
        return f"Mean predicted uplift ({self.crossfit.sample_.target_name})"

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
        result.outputs -= self.expected_output()
        return result


__tracker.validate()
