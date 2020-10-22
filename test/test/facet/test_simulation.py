import logging

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import approx

from sklearndf import TransformerDF
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.simulation import UnivariateUpliftSimulator
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
def uplift_simulator(
    crossfit: LearnerCrossfit, n_jobs: int
) -> UnivariateUpliftSimulator:
    return UnivariateUpliftSimulator(
        crossfit=crossfit,
        percentile_lower=10,
        percentile_upper=90,
        n_jobs=n_jobs,
        verbose=50,
    )


def test_actuals_simulation(uplift_simulator: UnivariateUpliftSimulator) -> None:

    assert_series_equal(
        uplift_simulator.simulate_actuals(),
        pd.Series(
            index=pd.RangeIndex(10, name="split_id"),
            data=(
                [0.0155072, 0.0280131, -0.0342100, 0.0155648, 0.0131959]
                + [-0.0486192, -0.0378004, 0.0068394, -0.0034286, 0.0202970]
            ),
            name="value",
        ),
        check_less_precise=True,
    )


def test_univariate_uplift_simulation(
    uplift_simulator: UnivariateUpliftSimulator,
) -> None:

    parameterized_feature = "LSTAT"
    partitioner = ContinuousRangePartitioner(max_partitions=10)

    sample = uplift_simulator.crossfit.sample_

    absolute_target_change_sr: pd.Series = (
        uplift_simulator._simulate_feature_with_values(
            feature_name=parameterized_feature,
            simulation_values=partitioner.fit(
                values=sample.features.loc[:, parameterized_feature]
            ).partitions_,
        )
    )

    # test aggregated values
    # the values on the right were computed from correct runs
    assert absolute_target_change_sr.min() == approx(-4.087962)
    assert absolute_target_change_sr.mean() == approx(-0.557341)
    assert absolute_target_change_sr.max() == approx(4.408145)

    aggregated_results = uplift_simulator._aggregate_simulation_results(
        results_per_split=absolute_target_change_sr
    )

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(data=[5.0, 10.0, 15.0, 20.0, 25.0])
    expected_data = {
        "percentile_10": [
            1.4028263372030223,
            -1.2328655768628771,
            -2.429197093534011,
            -2.7883113485337208,
            -2.7883113485337208,
        ],
        "percentile_50": [
            2.823081148237401,
            -0.6058660149256365,
            -1.2271310084526288,
            -1.2290430384093156,
            -1.2290430384093156,
        ],
        "percentile_90": [
            4.344860956307991,
            -0.2709598031049908,
            -0.63686624357766,
            -0.7253268617768278,
            -0.7253268617768278,
        ],
    }

    expected_df = pd.DataFrame(data=expected_data, index=index)
    assert_frame_equal(aggregated_results, expected_df)

    SimulationDrawer(style="text").draw(
        data=uplift_simulator.simulate_feature(
            name=parameterized_feature, partitioner=partitioner
        )
    )
