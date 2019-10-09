#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Univariate simulation of target uplift.
"""
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd

from gamma.common import ListLike
from gamma.ml.crossfit import ClassifierCrossfit, LearnerCrossfit, RegressorCrossfit
from gamma.sklearndf.transformation import FunctionTransformerDF
from gamma.yieldengine.partition import Partitioning, T_Number

_T_CrossFit = TypeVar("_T_CrossFit", bound=LearnerCrossfit)


class UnivariateSimulation:
    """
    Summary result of a univariate simulation.

    :param feature_name: name of the feature on which the simulation is made
    :param target_name: name of the target
    :param partitioning: the partition of ``feature_name`` used for the simulation
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
        feature_name: str,
        target_name: str,
        partitioning: Partitioning,
        median_change: ListLike[T_Number],
        min_change: ListLike[T_Number],
        max_change: ListLike[T_Number],
        min_percentile: float,
        max_percentile: float,
    ):
        self._feature_name = feature_name
        self._target_name = target_name
        self._partitioning = partitioning
        self._median_change = median_change
        self._min_change = min_change
        self._max_change = max_change
        self._min_percentile = min_percentile
        self._max_percentile = max_percentile

    @property
    def feature_name(self) -> str:
        """Name of the feature on which the simulation is made."""
        return self._feature_name

    @property
    def target_name(self) -> str:
        """Name of the target."""
        return self._target_name

    @property
    def partitioning(self) -> Partitioning:
        """The partition of ``feature_name`` used for the simulation."""
        return self._partitioning

    @property
    def median_change(self) -> ListLike[T_Number]:
        """Median average change determined by a simulation."""
        return self._median_change

    @property
    def min_change(self) -> ListLike[T_Number]:
        """
        Minimum average change, at the lower end of the confidence interval,
        determined by a simulation.
        """
        return self._min_change

    @property
    def max_change(self) -> ListLike[T_Number]:
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


class UnivariateSimulator(Generic[_T_CrossFit], ABC):
    """
    Estimates the average change in outcome for a range of values for a given feature,
    using cross-validated crossfit for all observations in a given data sample.

    Determines confidence intervals for the predicted changes using multiple
    crossfit for individual data points from different cross-validation splits.

    Works both with estimating the average change for target variables of regressors,
    or probabilities of binary classifiers.

    :param crossfit: cross-validated crossfit of a model for all observations \
        in a given sample
    :param min_percentile: lower bound of the confidence interval (default: 2.5)
    :param max_percentile: upper bound of the confidence interval (default: 97.5)
    """

    def __init__(
        self,
        crossfit: _T_CrossFit,
        min_percentile: float = 2.5,
        max_percentile: float = 97.5,
    ):

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

    @property
    def crossfit(self) -> _T_CrossFit:
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
    def simulate_feature(self, name: str, partitioning: Partitioning):
        """
        Simulate the average impact on the target when fixing the value of the given
        feature across all observations.

        :param name: the feature to run the simulation for
        :param partitioning: the partitioning of feature values to run simulations for
        """
        pass


class UnivariateProbabilitySimulator(UnivariateSimulator[ClassifierCrossfit]):
    """
    Univariate simulation for change in average predicted probability (CAPP) based on a
    classification model.
    """

    def simulate_feature(self, name: str, partitioning: Partitioning):
        """
        Simulate the average change in probability when fixing the
        value of the given feature across all observations.

        :param name: the feature to run the simulation for
        :param partitioning: the partitioning of feature values to run simulations for
        """
        raise NotImplementedError(
            "simulation of average change in probability will be included in a future "
            "release"
        )


class UnivariateUpliftSimulator(UnivariateSimulator[RegressorCrossfit]):
    """
    Univariate simulation for target uplift based on a regression model.
    """

    _COL_SPLIT_ID = "split_id"
    _COL_PARAMETER_VALUE = "parameter_value"
    _COL_ABSOLUTE_TARGET_CHANGE = "absolute_target_change"

    def simulate_feature(
        self, name: str, partitioning: Partitioning
    ) -> UnivariateSimulation:
        """
        Simulate the average target uplift when fixing the value of the given feature
        across all observations.

        :param name: the feature to run the simulation for
        :param partitioning: the partitioning of feature values to run simulations for
        """

        sample = self.crossfit.training_sample
        target = sample.target

        if not isinstance(target, pd.Series):
            raise NotImplementedError("multi-target simulations are not supported")

        simulated_values = partitioning.partitions()
        predicted_change = self._aggregate_simulation_results(
            results_per_split=self._simulate_feature_with_values(
                feature_name=name, simulated_values=simulated_values
            )
        )
        return UnivariateSimulation(
            feature_name=name,
            target_name=target.name,
            partitioning=partitioning,
            median_change=predicted_change.iloc[:, 1].values,
            min_change=predicted_change.iloc[:, 0].values,
            max_change=predicted_change.iloc[:, 2].values,
            min_percentile=self._min_percentile,
            max_percentile=self._max_percentile,
        )

    def _simulate_feature_with_values(
        self, feature_name: str, simulated_values: Iterable[Any]
    ) -> pd.DataFrame:
        """
        Run a simulation on a feature.

        For each combination of split_id and feature value the uplift (in % as a
        number between -1 and 1) of the target is computed. It is the uplift between
        crossfit on the sample where the `feature_name` column is set to the
        given value, compared to the crossfit on the original sample.

        :param feature_name: name of the feature to use in the simulation
        :param simulated_values: values to use in the simulation
        :return: data frame with three columns: `split_id`, `parameter_value` and
          `relative_target_change`.
        """

        crossfit = self.crossfit
        sample = crossfit.training_sample

        if feature_name not in sample.features.columns:
            raise ValueError(f"Feature '{feature_name}' not in sample")

        def _simulate_values() -> Generator[Tuple[int, Any, float], None, None]:
            feature_dtype = sample.features.loc[:, feature_name].dtype
            for value in simulated_values:
                # replace the simulated column with a constant value
                synthetic_sample = sample.replace_features(
                    FunctionTransformerDF(
                        func=lambda x: (
                            x.assign(**{feature_name: value}).astype(
                                {feature_name: feature_dtype}
                            )
                        ),
                        validate=False,
                    ).fit_transform(X=sample.features, y=sample.target)
                )

                for (
                    split_id,
                    (predictions_for_split_hist, predictions_for_split_syn),
                ) in enumerate(
                    zip(
                        crossfit._predictions_oob(sample=sample),
                        crossfit._predictions_oob(sample=synthetic_sample),
                    )
                ):
                    absolute_target_change = predictions_for_split_syn.mean(
                        axis=0
                    ) - predictions_for_split_hist.mean(axis=0)

                    yield split_id, value, absolute_target_change

        return pd.DataFrame(
            data=_simulate_values(),
            columns=[
                UnivariateUpliftSimulator._COL_SPLIT_ID,
                UnivariateUpliftSimulator._COL_PARAMETER_VALUE,
                UnivariateUpliftSimulator._COL_ABSOLUTE_TARGET_CHANGE,
            ],
        )

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
