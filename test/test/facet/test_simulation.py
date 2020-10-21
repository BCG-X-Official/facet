import logging

import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pytest import approx

from sklearndf import TransformerDF
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.simulation import (
    COL_LOWER_BOUND,
    COL_MEDIAN,
    COL_OUTPUT,
    COL_UPPER_BOUND,
    IDX_PARTITION,
    IDX_SPLIT,
    UnivariateSimulationResult,
    UnivariateTargetSimulator,
    UnivariateUpliftSimulator,
)
from facet.simulation.partition import ContinuousRangePartitioner
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
        shuffle_features=False,
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
    assert values.min() == approx(18.221038)
    assert values.mean() == approx(21.751659)
    assert values.max() == approx(26.717145)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(data=[5.0, 10.0, 15.0, 20.0, 25.0], name=IDX_PARTITION)

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [23.711826, 21.076134, 19.879803, 19.520689, 19.520689],
            name=COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [25.132081, 21.703134, 21.081869, 21.079957, 21.079957],
            name=COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [26.653861, 22.038040, 21.672134, 21.583673, 21.583673],
            name=COL_UPPER_BOUND,
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
            index=pd.RangeIndex(10, name=IDX_SPLIT),
            data=(
                [0.345951, 0.624943, -0.76319, 0.3472346, 0.294388]
                + [-1.084646, -0.843289, 0.1525806, -0.0764880, 0.4528048]
            ),
            name=COL_OUTPUT,
        ),
        check_less_precise=True,
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
    assert values.min() == approx(-4.087962)
    assert values.mean() == approx(-0.557341)
    assert values.max() == approx(4.408145)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(data=[5.0, 10.0, 15.0, 20.0, 25.0], name=IDX_PARTITION)

    assert_series_equal(
        simulation_result.outputs_lower_bound(),
        pd.Series(
            [
                1.4028263372030223,
                -1.2328655768628771,
                -2.429197093534011,
                -2.7883113485337208,
                -2.7883113485337208,
            ],
            name=COL_LOWER_BOUND,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_median(),
        pd.Series(
            [
                2.823081148237401,
                -0.6058660149256365,
                -1.2271310084526288,
                -1.2290430384093156,
                -1.2290430384093156,
            ],
            name=COL_MEDIAN,
            index=index,
        ),
    )

    assert_series_equal(
        simulation_result.outputs_upper_bound(),
        pd.Series(
            [
                4.344860956307991,
                -0.2709598031049908,
                -0.63686624357766,
                -0.7253268617768278,
                -0.7253268617768278,
            ],
            name=COL_UPPER_BOUND,
            index=index,
        ),
    )

    SimulationDrawer(style="text").draw(
        data=uplift_simulator.simulate_feature(
            feature_name=parameterized_feature, partitioner=partitioner
        )
    )
