import logging

import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pytest import approx

from sklearndf import TransformerDF
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.data.partition import ContinuousRangePartitioner
from facet.simulation import (
    UnivariateProbabilitySimulator,
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

    absolute_target_change_df: pd.DataFrame = simulation_result.outputs

    values = absolute_target_change_df.values

    # test aggregated values
    # the values on the right were computed from correct runs
    assert values.min() == approx(18.472759)
    assert values.mean() == approx(22.081310)
    assert values.max() == approx(28.471793)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[5.0, 10.0, 15.0, 20.0, 25.0], name=UnivariateTargetSimulator.IDX_PARTITION
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [22.431173, 19.789556, 18.853876, 18.853876, 18.853876],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [25.782475, 22.310836, 21.302304, 21.011027, 21.011027],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [
                27.750435,
                23.621475,
                23.031676,
                22.906156,
                22.906156,
            ],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
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
                [
                    3.207810,
                    1.807740,
                    0.709917,
                    -2.392966,
                    1.530005,
                    -2.394199,
                    1.389225,
                    -3.261376,
                    2.248752,
                    1.226377,
                ]
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
    assert values.min() == approx(-3.836241)
    assert values.mean() == approx(-0.2276897)
    assert values.max() == approx(6.162793)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[5.0, 10.0, 15.0, 20.0, 25.0], name=UnivariateUpliftSimulator.IDX_PARTITION
    )

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [0.122173, -2.519444, -3.455124, -3.455124, -3.455124],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [3.473475, 0.00183626, -1.006696, -1.297973, -1.297973],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [5.441435, 1.312475, 0.722676, 0.597156, 0.597156],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    SimulationDrawer(style="text").draw(data=simulation_result)


def test_univariate_probability_simulation(
    iris_classifier_crossfit_binary: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> None:
    parameterized_feature = "sepal length (cm)"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    print(iris_classifier_crossfit_binary.sample_.feature_names)

    proba_simulator = UnivariateProbabilitySimulator(
        crossfit=iris_classifier_crossfit_binary,
        confidence_level=0.95,
        n_jobs=n_jobs,
        verbose=50,
    )

    simulation_result: UnivariateSimulationResult = proba_simulator.simulate_feature(
        feature_name=parameterized_feature, partitioner=partitioner
    )

    index = pd.Index(
        data=[5, 5.5, 6, 6.5, 7, 7.5], name=UnivariateUpliftSimulator.IDX_PARTITION
    )

    assert simulation_result.baseline == approx(0.5)

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [0.346255, 0.346255, 0.353697, 0.394167, 0.401895, 0.417372],
            name=UnivariateSimulationResult.COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [0.460432, 0.450516, 0.469412, 0.488569, 0.492651, 0.507788],
            name=UnivariateSimulationResult.COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [0.582565, 0.562096, 0.570590, 0.580023, 0.599714, 0.602303],
            name=UnivariateSimulationResult.COL_UPPER_BOUND,
            index=index,
        ),
    )

    SimulationDrawer(style="text").draw(data=simulation_result)
