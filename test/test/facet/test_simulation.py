import logging

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from pytest import approx

from sklearndf import TransformerDF
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.data.partition import ContinuousRangePartitioner
from facet.simulation import (
    UnivariateSimulationResult,
    UnivariateTargetSimulator,
    UnivariateUpliftSimulator,
)
from facet.simulation.viz import SimulationDrawer
from facet.validation import StationaryBootstrapCV

log = logging.getLogger(__name__)

N_SPLITS = 10


@pytest.fixture
def crossfit(
    sample: Sample, simple_preprocessor: TransformerDF, n_jobs: int
) -> LearnerCrossfit:
    # use a pre-optimised model
    return LearnerCrossfit(
        pipeline=RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=LGBMRegressorDF(
                max_depth=10, min_split_gain=0.2, num_leaves=50, random_state=42
            ),
        ),
        cv=StationaryBootstrapCV(n_splits=N_SPLITS, random_state=42),
        n_jobs=n_jobs,
    ).fit(sample=sample)


@pytest.fixture
def subsample() -> pd.Index:
    return pd.Index(
        [8, 77, 65, 43, 43, 85, 8, 69, 20, 9, 52, 97, 73, 76, 71, 78]
        + [51, 12, 83, 45, 50, 37, 18, 92, 78, 64, 40, 82, 54, 44, 45, 22]
        + [9, 55, 88, 6, 85, 82, 27, 63, 16, 75, 70, 35, 6, 97, 44, 89, 67, 77]
    )


@pytest.fixture
def target_simulator(
    crossfit: LearnerCrossfit, n_jobs: int
) -> UnivariateTargetSimulator:
    return UnivariateTargetSimulator(
        crossfit=crossfit,
        confidence_level=0.8,
        n_jobs=n_jobs,
        verbose=50,
    )


@pytest.fixture
def uplift_simulator(
    crossfit: LearnerCrossfit, n_jobs: int
) -> UnivariateUpliftSimulator:
    return UnivariateUpliftSimulator(
        crossfit=crossfit,
        confidence_level=0.8,
        n_jobs=n_jobs,
        verbose=50,
    )


def test_univariate_target_simulation(
    target_simulator: UnivariateTargetSimulator,
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    simulation_result: UnivariateSimulationResult = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    values = simulation_result.outputs.values

    # test aggregated values
    # the values on the right were computed from correct runs
    assert values.min() == approx(18.47276)
    assert values.mean() == approx(22.63754)
    assert values.max() == approx(28.47179)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        name=UnivariateTargetSimulator.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [22.431173, 22.431173, 19.789556, 18.853876, 18.853876, 18.853876],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [25.782475, 25.782475, 22.310836, 21.302304, 21.011027, 21.011027],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [27.750435, 27.750435, 23.621475, 23.031676, 22.906156, 22.906156],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 31, 37, 19, 8, 1]
    )

    SimulationDrawer(style="text").draw(
        data=target_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_target_subsample_simulation(
    crossfit: LearnerCrossfit, subsample: pd.Index, n_jobs: int
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    sample_index = crossfit.sample_.index

    with pytest.raises(
        ValueError,
        match=(
            "arg subsample includes indices not contained in the simulation sample: "
            r"\[-1, 9999\]"
        ),
    ):
        UnivariateTargetSimulator(
            crossfit=crossfit,
            subsample=pd.Index([*sample_index, -1, 9999]),
        ).simulate_feature(
            feature_name=parameterized_feature,
            partitioner=partitioner,
        )

    target_simulator = UnivariateTargetSimulator(
        crossfit=crossfit,
        subsample=subsample,
        confidence_level=0.8,
        n_jobs=n_jobs,
        verbose=50,
    )

    simulation_result: UnivariateSimulationResult = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    values = simulation_result.outputs.values

    # test aggregated values
    # the values on the right were computed from correct runs
    assert values.min() == approx(17.92365)
    assert values.mean() == approx(23.30506)
    assert values.max() == approx(28.60988)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        name=UnivariateTargetSimulator.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [22.233849, 22.233849, 22.233849, 20.942154, 19.444643]
            + [19.363522, 18.300420, 18.300420, 18.300420],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [25.913666, 25.913666, 25.913666, 24.445583, 22.575495]
            + [22.403473, 22.288344, 21.642255, 21.430772],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [28.230187, 28.230187, 28.230187, 25.805393, 24.296859]
            + [24.221809, 24.174851, 23.640126, 23.640126],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 4, 9, 10, 10, 6, 2, 1, 4]
    )

    SimulationDrawer(style="text").draw(
        data=target_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_uplift_subsample_simulation_full_sample(
    crossfit: LearnerCrossfit, subsample: pd.Index, n_jobs: int
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    target_simulator = UnivariateUpliftSimulator(
        crossfit=crossfit,
        subsample=subsample,
        confidence_level=0.95,
        n_jobs=n_jobs,
        verbose=50,
    )

    target_simulator.full_sample = True

    simulation_result: UnivariateSimulationResult = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        name=UnivariateTargetSimulator.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound().round(6),
        pd.Series(
            [1.800835, 1.800835, 1.800835, -0.320393, -1.847194]
            + [-2.074327, -2.539217, -2.825394, -2.825394],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median().round(6),
        pd.Series(
            [2.696227, 2.696227, 2.696227, 0.652706, -0.878943]
            + [-1.081172, -1.494944, -1.750046, -1.750046],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound().round(6),
        pd.Series(
            [3.59162, 3.59162, 3.59162, 1.625805, 0.089307]
            + [-0.088017, -0.450671, -0.674698, -0.674698],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 4, 9, 10, 10, 6, 2, 1, 4]
    )

    SimulationDrawer(style="text").draw(
        data=target_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_actuals_simulation(uplift_simulator: UnivariateUpliftSimulator) -> None:

    assert_series_equal(
        uplift_simulator.simulate_actuals(),
        pd.Series(
            index=pd.RangeIndex(10, name=UnivariateUpliftSimulator.IDX_SPLIT),
            data=(
                [3.207810, 1.807740, 0.709917, -2.392966, 1.530005]
                + [-2.394199, 1.389225, -3.261376, 2.248752, 1.226377]
            ),
            name=UnivariateUpliftSimulator.COL_OUTPUT,
        ),
    )


def test_univariate_uplift_simulation(
    uplift_simulator: UnivariateUpliftSimulator,
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    simulation_result: UnivariateSimulationResult = uplift_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    absolute_target_change_df: pd.DataFrame = simulation_result.outputs

    values = absolute_target_change_df.values

    # test aggregated values
    # the values on the right were computed from correct runs
    assert values.min() == approx(-3.83624)
    assert values.mean() == approx(0.3285436)
    assert values.max() == approx(6.16279)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        name=UnivariateUpliftSimulator.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [0.122173, 0.122173, -2.519444, -3.455124, -3.455124, -3.455124],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [3.473475, 3.473475, 0.00183626, -1.006696, -1.297973, -1.297973],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [5.441435, 5.441435, 1.312475, 0.722676, 0.597156, 0.597156],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 31, 37, 19, 8, 1]
    )

    SimulationDrawer(style="text").draw(
        data=uplift_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_uplift_subsample_simulation(
    crossfit: LearnerCrossfit, subsample: pd.Index, n_jobs: int
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    sample_index = crossfit.sample_.index

    with pytest.raises(
        ValueError,
        match=(
            "arg subsample includes indices not contained in the simulation sample: "
            r"\[-1, 9999\]"
        ),
    ):
        UnivariateUpliftSimulator(
            crossfit=crossfit, subsample=pd.Index([*sample_index, -1, 9999])
        ).simulate_feature(
            feature_name=parameterized_feature,
            partitioner=partitioner,
        )

    uplift_simulator = UnivariateUpliftSimulator(
        crossfit=crossfit,
        subsample=subsample,
        confidence_level=0.8,
        n_jobs=n_jobs,
        verbose=50,
    )

    simulation_result: UnivariateSimulationResult = uplift_simulator.simulate_feature(
        feature_name=parameterized_feature, partitioner=partitioner
    )

    absolute_target_change_df: pd.DataFrame = simulation_result.outputs

    values = absolute_target_change_df.values

    # test aggregated values
    # the values on the right were computed from correct runs
    assert values.min() == approx(-5.02235)
    assert values.mean() == approx(0.359062)
    assert values.max() == approx(5.66388)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        name=UnivariateUpliftSimulator.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [-0.712151, -0.712151, -0.712151, -2.003846, -3.501357]
            + [-3.582478, -4.64558, -4.64558, -4.64558],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [2.967666, 2.967666, 2.967666, 1.499583, -0.370505]
            + [-0.542527, -0.657656, -1.303745, -1.515228],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [5.284187, 5.284187, 5.284187, 2.859393, 1.350859]
            + [1.275809, 1.228851, 0.694126, 0.694126],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 4, 9, 10, 10, 6, 2, 1, 4]
    )

    SimulationDrawer(style="text").draw(
        data=uplift_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )
