import logging

import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal, assert_series_equal

# noinspection PyPackageRequirements
from pytest import approx

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.validation import StationaryBootstrapCV
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression.extra import LGBMRegressorDF
from gamma.yieldengine.partition import ContinuousRangePartitioner
from gamma.yieldengine.simulation import UnivariateUpliftSimulator

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
        min_percentile=10,
        max_percentile=90,
        n_jobs=n_jobs,
        verbose=50,
    )


def test_actuals_simulation(uplift_simulator: UnivariateUpliftSimulator) -> None:

    assert_series_equal(
        uplift_simulator.simulate_actuals(),
        pd.Series(
            index=pd.RangeIndex(10, name="crossfit_id"),
            data=[
                0.0022346580154160023,
                0.002617709072003871,
                0.0007107764530731586,
                -0.005023988425882142,
                0.004010384123196653,
                0.006379100635514501,
                0.0005625736529850656,
                -0.0018641287749763258,
                0.0038418237873085737,
                -0.0025806445857182725,
            ],
            name="relative deviations of mean predictions from mean targets",
        ),
    )


def test_univariate_uplift_simulation(
    uplift_simulator: UnivariateUpliftSimulator
) -> None:

    parameterized_feature = "Step4-6 RawMat Vendor Compound08 Purity (#)"

    sample = uplift_simulator.crossfit.training_sample

    res = uplift_simulator._simulate_feature_with_values(
        feature_name=parameterized_feature,
        simulated_values=ContinuousRangePartitioner()
        .fit(values=sample.features.loc[:, parameterized_feature])
        .partitions(),
    )

    log.debug(res)
    # test aggregated values
    # the values on the right were computed from correct runs
    absolute_target_change_sr = res.loc[
        :, UnivariateUpliftSimulator._COL_ABSOLUTE_TARGET_CHANGE
    ]
    assert absolute_target_change_sr.min() == approx(-0.6453681553897965)
    assert absolute_target_change_sr.mean() == approx(-0.05250827061702283)
    assert absolute_target_change_sr.max() == approx(0.8872351297966503)

    aggregated_results = uplift_simulator._aggregate_simulation_results(
        results_per_crossfit=res
    )
    log.debug(aggregated_results)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[29.0, 29.5, 30.0, 30.5, 31.0],
        name=UnivariateUpliftSimulator._COL_PARAMETER_VALUE,
    )
    expected_data = {
        "percentile_10": [
            -0.47985361959947925,
            -0.1236605280451073,
            -0.01637965080466941,
            -0.01637965080466941,
            -0.0006339046007731496,
        ],
        "percentile_50": [
            -0.2851650197904405,
            0.39945412023171656,
            0.39945412023171656,
            0.39945412023171656,
            0.41147488809989596,
        ],
        "percentile_90": [
            -0.09558476689426461,
            0.8104729183863626,
            0.8104729183863626,
            0.8104729183863626,
            0.8093686301666558,
        ],
    }

    expected_df = pd.DataFrame(data=expected_data, index=index)
    assert_frame_equal(aggregated_results.loc[index], expected_df)

    log.debug(f"\n{aggregated_results}")
