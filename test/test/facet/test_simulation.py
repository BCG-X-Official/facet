import logging

import numpy as np
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


@pytest.fixture  # type: ignore
def model(
    sample: Sample, simple_preprocessor: TransformerDF
) -> RegressorPipelineDF[LGBMRegressorDF]:
    # use a pre-optimised model
    return RegressorPipelineDF(
        preprocessing=simple_preprocessor,
        regressor=LGBMRegressorDF(
            max_depth=10, min_split_gain=0.2, num_leaves=50, random_state=42
        ),
    ).fit(X=sample.features, y=sample.target)


@pytest.fixture  # type: ignore
def subsample(sample: Sample) -> Sample:
    return sample.subsample(
        iloc=(
            [8, 77, 65, 43, 43, 85, 8, 69, 20, 9, 52, 97, 73, 76, 71, 78]
            + [51, 12, 83, 45, 50, 37, 18, 92, 78, 64, 40, 82, 54, 44, 45, 22]
            + [9, 55, 88, 6, 85, 82, 27, 63, 16, 75, 70, 35, 6, 97, 44, 89, 67, 77]
        )
    )


@pytest.fixture  # type: ignore
def target_simulator(
    model: RegressorPipelineDF[LGBMRegressorDF], sample: Sample, n_jobs: int
) -> UnivariateTargetSimulator:
    return UnivariateTargetSimulator(
        model=model, sample=sample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )


@pytest.fixture  # type: ignore
def uplift_simulator(
    model: RegressorPipelineDF[LGBMRegressorDF], sample: Sample, n_jobs: int
) -> UnivariateUpliftSimulator:
    return UnivariateUpliftSimulator(
        model=model, sample=sample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )


def test_univariate_target_simulation(
    target_simulator: UnivariateTargetSimulator,
) -> None:
    parameterized_feature = "HouseAge"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    simulation_result: UnivariateSimulationResult[
        np.float_
    ] = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test simulation results

    index = pd.Index(
        data=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND].round(
            4
        ),
        pd.Series(
            [1.4621, 1.4621, 1.6542, 1.9865, 2.2322, 2.2322],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN].round(4),
        pd.Series(
            [1.5430, 1.5430, 1.7444, 2.0929, 2.3398, 2.3398],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND].round(
            5
        ),
        pd.Series(
            [1.62397, 1.62397, 1.8346, 2.19933, 2.44734, 2.44734],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 9, 23, 32, 23, 12]
    )

    SimulationDrawer(style="text").draw(
        data=target_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_target_subsample_simulation_80(
    model: RegressorPipelineDF[LGBMRegressorDF], subsample: Sample, n_jobs: int
) -> None:
    parameterized_feature = "HouseAge"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    target_simulator = UnivariateTargetSimulator(
        model=model, sample=subsample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )

    simulation_result: UnivariateSimulationResult[
        np.float_
    ] = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
        lower_bound=3.8,
    )

    # test simulation results

    index = pd.Index(
        data=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND].round(
            5
        ),
        pd.Series(
            [1.63804, 1.63804, 1.86338, 2.23901, 2.44907, 2.44907],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN].round(5),
        pd.Series(
            [1.74114, 1.74114, 1.97514, 2.36965, 2.58642, 2.58642],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND].round(
            5
        ),
        pd.Series(
            [1.84423, 1.84423, 2.08689, 2.50029, 2.72377, 2.72377],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(simulation_result.partitioner.frequencies_, [2, 5, 18, 8, 11, 6])

    SimulationDrawer(style="text").draw(
        data=target_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_uplift_subsample_simulation_95(
    model: RegressorPipelineDF[LGBMRegressorDF], subsample: Sample, n_jobs: int
) -> None:
    parameterized_feature = "HouseAge"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    target_simulator = UnivariateUpliftSimulator(
        model=model, sample=subsample, confidence_level=0.95, n_jobs=n_jobs, verbose=50
    )

    simulation_result: UnivariateSimulationResult[
        np.float_
    ] = target_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test simulation results

    index = pd.Index(
        data=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND].round(
            6
        ),
        pd.Series(
            [-0.47617, -0.47617, -0.255413, 0.110223, 0.316729, 0.316729],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN].round(6),
        pd.Series(
            [-0.318501, -0.318501, -0.084503, 0.310012, 0.526783, 0.526783],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND].round(
            6
        ),
        pd.Series(
            [-0.160831, -0.160831, 0.086407, 0.509801, 0.736836, 0.736836],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(simulation_result.partitioner.frequencies_, [2, 5, 18, 8, 11, 6])

    SimulationDrawer(style="text").draw(
        data=target_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_uplift_simulation(
    uplift_simulator: UnivariateUpliftSimulator,
) -> None:
    parameterized_feature = "HouseAge"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    simulation_result: UnivariateSimulationResult[
        np.float_
    ] = uplift_simulator.simulate_feature(
        feature_name=parameterized_feature,
        partitioner=partitioner,
    )

    # test simulation results

    index = pd.Index(
        data=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND].round(
            5
        ),
        pd.Series(
            [-0.48552, -0.48552, -0.2934, 0.03889, 0.28455, 0.28455],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN].round(5),
        pd.Series(
            [-0.40459, -0.40459, -0.20322, 0.14529, 0.39212, 0.39212],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND].round(
            5
        ),
        pd.Series(
            [-0.32367, -0.32367, -0.11304, 0.25169, 0.4997, 0.4997],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(
        simulation_result.partitioner.frequencies_, [1, 9, 23, 32, 23, 12]
    )

    SimulationDrawer(style="text").draw(
        data=uplift_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )


def test_univariate_uplift_subsample_simulation(
    model: RegressorPipelineDF[LGBMRegressorDF], subsample: Sample, n_jobs: int
) -> None:
    parameterized_feature = "HouseAge"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    uplift_simulator = UnivariateUpliftSimulator(
        model=model, sample=subsample, confidence_level=0.8, n_jobs=n_jobs, verbose=50
    )

    simulation_result: UnivariateSimulationResult[
        np.float_
    ] = uplift_simulator.simulate_feature(
        feature_name=parameterized_feature, partitioner=partitioner
    )

    # test simulation results

    index = pd.Index(
        data=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        name=UnivariateSimulationResult.IDX_PARTITION,
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_LOWER_BOUND].round(
            5
        ),
        pd.Series(
            [-0.4216, -0.4216, -0.19626, 0.17938, 0.38944, 0.38944],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_MEAN].round(5),
        pd.Series(
            [-0.3185, -0.3185, -0.0845, 0.31001, 0.52678, 0.52678],
            name=UnivariateSimulationResult.COL_MEAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.data.loc[:, UnivariateSimulationResult.COL_UPPER_BOUND].round(
            5
        ),
        pd.Series(
            [-0.21541, -0.21541, 0.02725, 0.44065, 0.66413, 0.66413],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    assert_array_equal(simulation_result.partitioner.frequencies_, [2, 5, 18, 8, 11, 6])

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

    simulation_result: UnivariateSimulationResult[
        np.float_
    ] = proba_simulator.simulate_feature(
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
