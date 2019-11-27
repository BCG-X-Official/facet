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
        base_learner=RegressorPipelineDF(
            preprocessing=simple_preprocessor,
            regressor=LGBMRegressorDF(
                max_depth=10, min_split_gain=0.2, num_leaves=50, random_state=42
            ),
        ),
        cv=StationaryBootstrapCV(n_splits=N_SPLITS, random_state=42),
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
    assert absolute_target_change_sr.min() == approx(-0.7316366998749118)
    assert absolute_target_change_sr.mean() == approx(-0.09974330639733585)
    assert absolute_target_change_sr.max() == approx(0.6741931812285031)

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
            -0.6818825099675685,
            -0.007029585576065488,
            0.0538630336704145,
            0.0538630336704145,
            0.06007108162896415,
        ],
        "percentile_50": [
            -0.3355583717195465,
            0.3669534874333582,
            0.3669534874333582,
            0.3669534874333582,
            0.37881715617286993,
        ],
        "percentile_90": [
            -0.011813212687955503,
            0.5806558851769821,
            0.5806558851769821,
            0.5806558851769821,
            0.5790936685270693,
        ],
    }

    expected_df = pd.DataFrame(data=expected_data, index=index)
    assert_frame_equal(aggregated_results.loc[index], expected_df)

    log.debug(f"\n{aggregated_results}")
