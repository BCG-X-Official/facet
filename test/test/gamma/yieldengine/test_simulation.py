import logging
from typing import *

import pandas as pd
from pandas.util.testing import assert_frame_equal

# noinspection PyPackageRequirements
from pytest import approx

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.selection import ParameterGrid
from gamma.ml.validation import StationaryBootstrapCV
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression.extra import LGBMRegressorDF
from gamma.yieldengine.partition import ContinuousRangePartitioner
from gamma.yieldengine.simulation import UnivariateUpliftSimulator

log = logging.getLogger(__name__)

N_SPLITS = 10


def test_univariate_simulation(
    batch_table: pd.DataFrame,
    regressor_grids: Iterable[ParameterGrid],
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs,
) -> None:

    # use a pre-optimised model
    crossfit = LearnerCrossfit(
        pipeline=RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=LGBMRegressorDF(
                max_depth=10, min_split_gain=0.2, num_leaves=50, random_state=42
            ),
        ),
        cv=StationaryBootstrapCV(n_splits=N_SPLITS, random_state=42),
        shuffle_features=False,
    ).fit(sample=sample)

    simulator = UnivariateUpliftSimulator(
        crossfit=crossfit,
        min_percentile=10,
        max_percentile=90,
        n_jobs=n_jobs,
        verbose=50,
    )

    parameterized_feature = "Step4-6 RawMat Vendor Compound08 Purity (#)"

    res = simulator._simulate_feature_with_values(
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
    assert absolute_target_change_sr.min() == approx(-0.7282927117948859)
    assert absolute_target_change_sr.mean() == approx(-0.0870004327894347)
    assert absolute_target_change_sr.max() == approx(0.804310573391561)

    aggregated_results = simulator._aggregate_simulation_results(results_per_split=res)
    log.debug(aggregated_results)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run

    index = pd.Index(
        data=[29.0, 29.5, 30.0, 30.5, 31.0],
        name=UnivariateUpliftSimulator._COL_PARAMETER_VALUE,
    )
    expected_data = {
        "percentile_10": [
            -0.6004228916492564,
            0.027750755748968466,
            0.052682663898563936,
            0.052682663898563936,
            0.13982249635328545,
        ],
        "percentile_50": [
            -0.25439493925730616,
            0.2918181523488208,
            0.2918181523488208,
            0.2918181523488208,
            0.30152469018852557,
        ],
        "percentile_90": [
            -0.0976467669962471,
            0.6926482832226767,
            0.6926482832226767,
            0.6926482832226767,
            0.69154399500297,
        ],
    }

    expected_df = pd.DataFrame(data=expected_data, index=index)
    assert_frame_equal(aggregated_results.loc[index], expected_df)

    log.debug(f"\n{aggregated_results}")
