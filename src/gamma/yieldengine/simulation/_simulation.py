"""
Core implementation of :mod:`gamma.yieldengine.simulation`
"""

from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd

from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.sklearndf import ClassifierDF, RegressorDF
from gamma.yieldengine.partition import Partitioner, T_Number

__all__ = [
    "UnivariateSimulation",
    "BaseUnivariateSimulator",
    "UnivariateUpliftSimulator",
    "UnivariateProbabilitySimulator",
]

T_CrossFit = TypeVar("T_CrossFit", bound=LearnerCrossfit)
T_RegressorDF = TypeVar("T_RegressorDF", bound=RegressorDF)
T_ClassifierDF = TypeVar("T_ClassifierDF", bound=ClassifierDF)


class UnivariateSimulation(Generic[T_Number]):
    """
    Summary result of a univariate simulation.

    :param feature: name of the feature on which the simulation is made
    :param target: name of the target
    :param partitioning: the partition of ``feature`` used for the simulation
    :param median_change: the median change values
    :param min_change:  the low percentile change values
    :param max_change: the high percentile change values
    :param min_percentile: the percentile used to compute ``min_uplift``. Must be a
      number between 0 and 100
    :param max_percentile: the percentile used to compute ``max_uplift``. Must be a
      number between 0 and 100
    """

    def __init__(
        self,
        feature: str,
        target: str,
        partitioning: Partitioner,
        median_change: Sequence[T_Number],
        min_change: Sequence[T_Number],
        max_change: Sequence[T_Number],
        min_percentile: float,
        max_percentile: float,
    ):
        self._feature = feature
        self._target = target
        self._partitioning = partitioning
        self._median_change = median_change
        self._min_change = min_change
        self._max_change = max_change
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    @property
    def feature(self) -> str:
        """Name of the feature on which the simulation is made."""
        return self._feature

    @property
    def target(self) -> str:
        """Name of the target."""
        return self._target

    @property
    def partitioner(self) -> Partitioner:
        """The partitioner generating the feature values to be simulated."""
        return self._partitioning

    @property
    def median_change(self) -> Sequence[T_Number]:
        """Median average change determined by a simulation."""
        return self._median_change

    @property
    def min_change(self) -> Sequence[T_Number]:
        """
        Minimum average change, at the lower end of the confidence interval,
        determined by a simulation.
        """
        return self._min_change

    @property
    def max_change(self) -> Sequence[T_Number]:
        """
        Minimum average change, at the lower end of the confidence interval,
        determined by a simulation.
        """
        return self._max_change

    @property
    def min_percentile(self) -> float:
        """
        Percentile of the lower end of thw confidence interval.
        """
        return self._min_percentile

    @property
    def max_percentile(self) -> float:
        """
        Percentile of the upper end of thw confidence interval.
        """
        return self._max_percentile


class BaseUnivariateSimulator(ParallelizableMixin, ABC, Generic[T_CrossFit]):
    """
    Estimates the average change in outcome for a range of values for a given feature,
    using cross-validated crossfit for all observations in a given data sample.

    Determines confidence intervals for the predicted changes using multiple
    crossfit for individual data points from different cross-validation splits.

    Works both with estimating the average change for target variables of regressors,
    or probabilities of binary classifiers.
    """

    def __init__(
        self,
        crossfit: T_CrossFit,
        *,
        min_percentile: float = 2.5,
        max_percentile: float = 97.5,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ):
        """
        :param crossfit: cross-validated crossfit of a model for all observations \
        in a given sample
        :param min_percentile: lower bound of the confidence interval (default: 2.5)
        :param max_percentile: upper bound of the confidence interval (default: 97.5)
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if not crossfit.is_fitted:
            raise ValueError("arg crossfit expected to be fitted")

        if not 0 <= min_percentile <= 100:
            raise ValueError(
                f"arg min_percentile={min_percentile} must be in the range"
                "from 0 to 100"
            )
        if not 0 <= max_percentile <= 100:
            raise ValueError(
                f"arg max_percentile={max_percentile} must be in the range"
                "from 0 to 1"
            )

        self._crossfit = crossfit
        self._max_percentile = max_percentile
        self._min_percentile = min_percentile

    # add parallelization parameters to __init__ docstring
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @property
    def crossfit(self) -> T_CrossFit:
        """The crossfit used for the simulation."""
        return self._crossfit

    @property
    def min_percentile(self) -> float:
        """
        Percentile of the lower end of the confidence interval.
        """
        return self._min_percentile

    @property
    def max_percentile(self) -> float:
        """
        Percentile of the upper end of the confidence interval.
        """
        return self._max_percentile

    @abstractmethod
    def simulate_feature(self, name: str, partitioner: Partitioner):
        """
        Simulate the average impact on the target when fixing the value of the given
        feature across all observations.

        :param name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for
        """
        pass


class UnivariateProbabilitySimulator(
    BaseUnivariateSimulator[LearnerCrossfit[T_ClassifierDF]], Generic[T_ClassifierDF]
):
    """
    Univariate simulation for change in average predicted probability (CAPP) based on a
    classification model.
    """

    def simulate_feature(self, name: str, partitioner: Partitioner):
        """
        Simulate the average change in probability when fixing the
        value of the given feature across all observations.

        :param name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for
        """
        raise NotImplementedError(
            "simulation of average change in probability will be included in a future "
            "release"
        )

    def _probabilities_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame]], None, None]:
        yield from self._classification_oob(
            sample=sample, method=lambda model, x: model.predict_proba(x)
        )

    def _log_probabilities_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame]], None, None]:
        yield from self._classification_oob(
            sample=sample, method=lambda model, x: model.predict_log_proba(x)
        )

    def _decision_function(
        self, sample: Sample
    ) -> Generator[Union[pd.Series, pd.DataFrame], None, None]:
        yield from self._classification_oob(
            sample=sample, method=lambda model, x: model.decision_function(x)
        )

    def _classification_oob(
        self,
        sample: Sample,
        method: Callable[
            [ClassifierDF, pd.DataFrame],
            Union[pd.DataFrame, List[pd.DataFrame], pd.Series],
        ],
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame], pd.Series], None, None]:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``COL_SPLIT_ID``),
        and with columns ``COL_PREDICTION` (the predicted value for the
        given observation and split), and ``COL_TARGET`` (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the crossfit per observation and test split
        """

        for split_id, (model, (_, test_indices)) in enumerate(
            zip(self.crossfit.models(), self.crossfit.splits())
        ):
            test_features = sample.features.iloc[test_indices, :]
            yield method(model, test_features)


class UnivariateUpliftSimulator(
    BaseUnivariateSimulator[LearnerCrossfit[T_RegressorDF]], Generic[T_RegressorDF]
):
    """
    Univariate simulation for target uplift based on a regression model.
    """

    _COL_SPLIT_ID = "split_id"
    _COL_PARAMETER_VALUE = "parameter_value"
    _COL_ABSOLUTE_TARGET_CHANGE = "absolute_target_change"

    def simulate_feature(
        self, name: str, partitioner: Partitioner
    ) -> UnivariateSimulation:
        """
        Simulate the average target uplift when fixing the value of the given feature
        across all observations.

        :param name: the feature to run the simulation for
        :param partitioner: the partitioner of feature values to run simulations for
        """

        sample = self.crossfit.training_sample
        target = sample.target

        if not isinstance(target, pd.Series):
            raise NotImplementedError("multi-target simulations are not supported")

        simulated_values = partitioner.fit(sample.features.loc[:, name]).partitions()
        predicted_change = self._aggregate_simulation_results(
            results_per_split=self._simulate_feature_with_values(
                feature_name=name, simulated_values=simulated_values
            )
        )
        return UnivariateSimulation(
            feature=name,
            target=target.name,
            partitioning=partitioner,
            median_change=predicted_change.iloc[:, 1].values,
            min_change=predicted_change.iloc[:, 0].values,
            max_change=predicted_change.iloc[:, 2].values,
            min_percentile=self._min_percentile,
            max_percentile=self._max_percentile,
        )

    def _simulate_feature_with_values(
        self, feature_name: str, simulated_values: Sequence[Any]
    ) -> pd.DataFrame:
        """
        Run a simulation on a feature.

        For each combination of split_id and feature value the uplift (in % as a
        number between -1 and 1) of the target is computed. It is the uplift between
        crossfit on the sample where the `feature` column is set to the
        given value, compared to the crossfit on the original sample.

        :param feature_name: name of the feature to use in the simulation
        :param simulated_values: values to use in the simulation
        :return: data frame with three columns: `split_id`, `parameter_value` and
          `relative_target_change`.
        """

        def _compact_data_frame(df: pd.DataFrame):
            # helper method to ensure a data frame is not a view, to minimise memory
            # requirements before passing on for parallel processing
            if df.values.base is df.values:
                return df
            else:
                return df.copy(deep=True)

        crossfit = self.crossfit
        sample = crossfit.training_sample

        if feature_name not in sample.features.columns:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        with self._parallel() as parallel:
            result: List[List[float]] = parallel(
                self._delayed(UnivariateUpliftSimulator._simulate_values_for_split)(
                    model=model,
                    subsample_features=_compact_data_frame(
                        sample.subsample(iloc=test_indices).features
                    ),
                    feature_name=feature_name,
                    simulated_values=simulated_values,
                )
                for (model, (_, test_indices)) in zip(
                    self.crossfit.models(), self.crossfit.splits()
                )
            )

        col_split_id = UnivariateUpliftSimulator._COL_SPLIT_ID
        col_parameter_value = UnivariateUpliftSimulator._COL_PARAMETER_VALUE
        col_absolute_target_change = (
            UnivariateUpliftSimulator._COL_ABSOLUTE_TARGET_CHANGE
        )

        return pd.concat(
            [
                pd.DataFrame(
                    {
                        col_split_id: split_id,
                        col_parameter_value: simulated_values,
                        col_absolute_target_change: absolute_target_changes,
                    }
                )
                for split_id, absolute_target_changes in enumerate(result)
            ]
        )

    @staticmethod
    def _simulate_values_for_split(
        model: T_RegressorDF,
        subsample_features: pd.DataFrame,
        feature_name: str,
        simulated_values: Sequence[Any],
    ) -> List[float]:
        # for a list of values to be simulated, return a list of absolute target changes
        feature_dtype = subsample_features.loc[:, feature_name].dtype

        def _absolute_target_change(value: Any) -> float:
            # replace the simulated column with a constant value
            synthetic_subsample_features = subsample_features.assign(
                **{feature_name: value}
            ).astype({feature_name: feature_dtype})

            predictions_for_split_hist = model.predict(X=subsample_features)
            predictions_for_split_syn = model.predict(X=synthetic_subsample_features)

            return predictions_for_split_syn.mean(
                axis=0
            ) - predictions_for_split_hist.mean(axis=0)

        return [_absolute_target_change(value) for value in simulated_values]

    def _aggregate_simulation_results(
        self, results_per_split: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aggregate uplift values computed by `simulate_feature`.

        For each parameter value, the percentile of uplift values (in the
        `relative_yield_change` column) are computed.

        :param results_per_split: data frame with columns `split_id`, `parameter_value`\
          and `relative_yield_change`
        :return: data frame with 3 columns `percentile_<min>`, `percentile_50`,
          `percentile<max>` where min/max are the min and max percentiles
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

            percentile_.__name__ = "percentile_%s" % n
            return percentile_

        return (
            results_per_split.drop(columns=UnivariateUpliftSimulator._COL_SPLIT_ID)
            .groupby(by=UnivariateUpliftSimulator._COL_PARAMETER_VALUE)[
                UnivariateUpliftSimulator._COL_ABSOLUTE_TARGET_CHANGE
            ]
            .agg(
                [
                    percentile(p)
                    for p in (self._min_percentile, 50, self._max_percentile)
                ]
            )
        )
