import logging
from typing import *

import pandas as pd
from pandas.util.testing import assert_frame_equal

# noinspection PyPackageRequirements
from pytest import approx

from gamma.ml import Sample
from gamma.ml.selection import ParameterGrid, RegressorRanker
from gamma.ml.validation import CircularCV
from gamma.sklearndf import TransformerDF
from gamma.yieldengine.partition import ContinuousRangePartitioning
from gamma.yieldengine.simulation import UnivariateUpliftSimulator

log = logging.getLogger(__name__)

N_SPLITS = 5
TEST_RATIO = 0.2


def test_univariate_simulation(
    batch_table: pd.DataFrame,
    regressor_grids: Iterable[ParameterGrid],
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs,
) -> None:

    # define the circular cross validator with just 5 splits (to speed up testing)
    circular_cv = CircularCV(test_ratio=TEST_RATIO, n_splits=N_SPLITS)

    model_ranker = RegressorRanker(
        grid=regressor_grids, sample=sample, cv=circular_cv, scoring="r2", n_jobs=n_jobs
    )

    predictions = model_ranker.best_model_predictions()

    simulator = UnivariateUpliftSimulator(
        predictions=predictions, min_percentile=10, max_percentile=90
    )

    parameterized_feature = "Step4-6 RawMat Vendor Compound08 Purity (#)"

    res = simulator._simulate_feature_with_values(
        feature_name=parameterized_feature,
        simulated_values=ContinuousRangePartitioning(
            values=sample.features.loc[:, parameterized_feature]
        ).partitions(),
    )

    log.debug(res)
    # test aggregated values
    # the values on the right were computed from correct runs
    assert res.iloc[:, 2].mean() == approx(-0.005341351699881821)
    assert res.iloc[:, 2].max() == approx(0.01904097474184785)
    assert res.iloc[:, 2].min() == approx(-0.050256813777029286)

    aggregated_results = simulator._aggregate_simulation_results(results_per_split=res)
    log.debug(aggregated_results)

    # test the first five rows of aggregated_results
    # the values were computed from a correct run
    dict_data = {
        "percentile_10": {
            24.0: -0.022085854777025404,
            24.5: -0.01544493841392831,
            25.0: -0.01544493841392831,
            25.5: -0.01544493841392831,
            26.0: -0.035765236211580544,
        },
        "percentile_50": {
            24.0: 0.011567498441119817,
            24.5: 0.0,
            25.0: 0.0,
            25.5: 0.0,
            26.0: 0.0,
        },
        "percentile_90": {
            24.0: 0.01860559941647595,
            24.5: 0.0,
            25.0: 0.0,
            25.5: 0.0,
            26.0: 0.0,
        },
    }
    index = pd.Index(
        data=[24.0, 24.5, 25.0, 25.5, 26.0],
        name=UnivariateUpliftSimulator.F_PARAMETER_VALUE,
    )
    df_test = pd.DataFrame(data=dict_data, index=index)
    assert_frame_equal(aggregated_results.head(), df_test)
