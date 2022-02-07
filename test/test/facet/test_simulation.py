import logging

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from pytest import approx

from sklearndf import TransformerDF
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from facet.data import Sample
from facet.data.partition import ContinuousRangePartitioner
from facet.simulation import (
    UnivariateProbabilitySimulator,
    UnivariateSimulationResult,
    UnivariateTargetSimulator,
    UnivariateUpliftSimulator,
)
from facet.simulation.viz import SimulationDrawer

log = logging.getLogger(__name__)

N_SPLITS = 10


@pytest.fixture
def model(sample: Sample, simple_preprocessor: TransformerDF) -> RegressorPipelineDF:
    # use a pre-optimised model
    return RegressorPipelineDF(
        preprocessing=simple_preprocessor,
        regressor=LGBMRegressorDF(
            max_depth=10, min_split_gain=0.2, num_leaves=50, random_state=42
        ),
    ).fit(X=sample.features, y=sample.target)


@pytest.fixture
def subsample(sample: Sample) -> Sample:
    return sample.subsample(
        iloc=(
            [8, 77, 65, 43, 43, 85, 8, 69, 20, 9, 52, 97, 73, 76, 71, 78]
            + [51, 12, 83, 45, 50, 37, 18, 92, 78, 64, 40, 82, 54, 44, 45, 22]
            + [9, 55, 88, 6, 85, 82, 27, 63, 16, 75, 70, 35, 6, 97, 44, 89, 67, 77]
        )
    )


@pytest.fixture
def target_simulator(
    model: RegressorPipelineDF, sample: Sample, n_jobs: int
) -> UnivariateTargetSimulator:
    return UnivariateTargetSimulator(
        model=model, sample=sample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )


@pytest.fixture
def uplift_simulator(
    model: RegressorPipelineDF, sample: Sample, n_jobs: int
) -> UnivariateUpliftSimulator:
    return UnivariateUpliftSimulator(
        model=model, sample=sample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
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

    # test simulation results

    index = pd.Index(
        data=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND],
        pd.Series(
            [24.98646, 24.98646, 21.15398, 20.23877, 20.23877, 20.23877],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN],
        pd.Series(
            [25.4571, 25.4571, 21.67744, 20.81063, 20.81063, 20.81063],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND],
        pd.Series(
            [25.92774, 25.92774, 22.2009, 21.38249, 21.38249, 21.38249],
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


def test_univariate_target_subsample_simulation_80(
    model: RegressorPipelineDF, subsample: Sample, n_jobs: int
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    target_simulator = UnivariateTargetSimulator(
        model=model, sample=subsample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )

    simulation_result: UnivariateSimulationResult = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test simulation results

    index = pd.Index(
        data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND],
        pd.Series(
            [25.05676, 25.05676, 25.05676, 22.96243, 21.43395]
            + [21.21544, 20.76824, 20.49282, 20.49282],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN],
        pd.Series(
            [25.642227, 25.642227, 25.642227, 23.598706, 22.067057]
            + [21.864828, 21.451056, 21.195954, 21.195954],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND],
        pd.Series(
            [26.22769, 26.22769, 26.22769, 24.23498, 22.70016]
            + [22.51422, 22.13387, 21.89909, 21.89909],
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


def test_univariate_uplift_subsample_simulation_95(
    model: RegressorPipelineDF, subsample: Sample, n_jobs: int
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    target_simulator = UnivariateUpliftSimulator(
        model=model, sample=subsample, confidence_level=0.95, n_jobs=n_jobs, verbose=50
    )

    simulation_result: UnivariateSimulationResult = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test simulation results

    index = pd.Index(
        data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND].round(
            6
        ),
        pd.Series(
            [1.800835, 1.800835, 1.800835, -0.320393, -1.847194]
            + [-2.074327, -2.539217, -2.825394, -2.825394],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN].round(6),
        pd.Series(
            [2.696227, 2.696227, 2.696227, 0.652706, -0.878943]
            + [-1.081172, -1.494944, -1.750046, -1.750046],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND].round(
            6
        ),
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


def test_univariate_uplift_simulation(
    uplift_simulator: UnivariateUpliftSimulator,
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    simulation_result: UnivariateSimulationResult = uplift_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test simulation results

    index = pd.Index(
        data=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND],
        pd.Series(
            [2.677461, 2.677461, -1.155017, -2.070234, -2.070234, -2.070234],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN],
        pd.Series(
            [3.148100, 3.148100, -0.631560, -1.498371, -1.498371, -1.498371],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND],
        pd.Series(
            [3.618739, 3.618739, -0.108103, -0.926508, -0.926508, -0.926508],
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
    model: RegressorPipelineDF, subsample: Sample, n_jobs: int
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    uplift_simulator = UnivariateUpliftSimulator(
        model=model, sample=subsample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )

    simulation_result: UnivariateSimulationResult = uplift_simulator.simulate_feature(
        feature_name=parameterized_feature, partitioner=partitioner
    )

    # test simulation results

    index = pd.Index(
        data=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND],
        pd.Series(
            [2.110762, 2.110762, 2.110762, 0.0164306, -1.512048]
            + [-1.730561, -2.177757, -2.453179, -2.453179],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN],
        pd.Series(
            [2.696227, 2.696227, 2.696227, 0.652706, -0.878943]
            + [-1.081172, -1.494944, -1.750046, -1.750046],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND],
        pd.Series(
            [3.281693, 3.281693, 3.281693, 1.288981, -0.245838]
            + [-0.431783, -0.81213, -1.046914, -1.046914],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 4, 9, 10, 10, 6, 2, 1, 4]
    )

    SimulationDrawer(style="text").draw(data=simulation_result)


def test_univariate_probability_simulation(
    iris_classifier_binary: ClassifierPipelineDF[RandomForestClassifierDF],
    iris_sample_binary: Sample,
    n_jobs: int,
) -> None:
    parameterized_feature = "sepal length (cm)"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    print(iris_sample_binary.feature_names)

    proba_simulator = UnivariateProbabilitySimulator(
        model=iris_classifier_binary,
        sample=iris_sample_binary,
        confidence_level=0.95,
        n_jobs=n_jobs,
        verbose=50,
    )

    simulation_result: UnivariateSimulationResult = proba_simulator.simulate_feature(
        feature_name=parameterized_feature, partitioner=partitioner
    )

    index = pd.Index(
        data=[5, 5.5, 6, 6.5, 7, 7.5, 8], name=UnivariateSimulationResult.IDX_PARTITION
    )

    assert simulation_result.baseline == approx(0.5)

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND],
        pd.Series(
            [0.415337, 0.390766, 0.401039, 0.420727, 0.425914, 0.452885, 0.452885],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN],
        pd.Series(
            [0.495814, 0.475288, 0.48689, 0.507294, 0.510055, 0.533888, 0.533888],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND],
        pd.Series(
            [0.576292, 0.559809, 0.57274, 0.593862, 0.594196, 0.614892, 0.614892],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    SimulationDrawer(style="text").draw(data=simulation_result)
