"""
Core implementation of :mod:`facet.simulation`
"""


import logging
from typing import Generic, Sequence, TypeVar

import numpy as np
import pandas as pd
from scipy import stats

from pytools.api import AllTracker

from facet.data.partition import Partitioner

log = logging.getLogger(__name__)

__all__ = [
    "UnivariateSimulationResult",
]


#
# Type variables
#

T_Values = TypeVar("T_Values", bound=np.generic)


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


class UnivariateSimulationResult(Generic[T_Values]):
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
    partitioner: Partitioner[T_Values]

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
        partitioner: Partitioner[T_Values],
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
            the standard error of the mean, ranging between 0.0 and 1.0 (exclusive)
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
