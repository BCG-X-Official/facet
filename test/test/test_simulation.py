import logging
from typing import *

import pandas as pd
from pandas.util.testing import assert_frame_equal
from pytest import approx

from gamma import Sample
from gamma.model.prediction import RegressorFitCV
from gamma.model.selection import ModelEvaluation, ModelGrid, ModelRanker
from gamma.model.validation import CircularCrossValidator
from gamma.sklearndf import TransformerDF
from gamma.yieldengine.partition import ContinuousRangePartitioning
from gamma.yieldengine.simulation import UnivariateSimulator

log = logging.getLogger(__name__)

N_SPLITS = 5
TEST_RATIO = 0.2


def test_univariate_simulation(
    batch_table: pd.DataFrame,
    regressor_grids: Iterable[ModelGrid],
    sample: Sample,
    simple_preprocessor: TransformerDF,
    available_cpus: int,
) -> None:

    # define the circular cross validator with just 5 splits (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=TEST_RATIO, num_splits=N_SPLITS)

    model_ranker: ModelRanker = ModelRanker(
        grids=regressor_grids, cv=circular_cv, scoring="r2"
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        sample=sample, n_jobs=available_cpus
    )

    model_fit = RegressorFitCV(
        model=model_ranking[0].model,
        cv=circular_cv,
        sample=sample,
        n_jobs=available_cpus,
    )

    sim = UnivariateSimulator(model_fit=model_fit)

    parameterized_feature = "Step4-6 RawMat Vendor Compound08 Purity (#)"

    res = sim.simulate_feature(
        feature_name=parameterized_feature,
        feature_values=ContinuousRangePartitioning(
            values=sample.features.loc[:, parameterized_feature]
        ).partitions(),
    )

    log.debug(res)
    # test aggregated values
    # the values on the right were computed from correct runs
    assert res.iloc[:, 2].mean() == approx(-0.005341351699881821)
    assert res.iloc[:, 2].max() == approx(0.01904097474184785)
    assert res.iloc[:, 2].min() == approx(-0.050256813777029286)

    aggregated_results = UnivariateSimulator.aggregate_simulation_results(
        results_per_split=res, percentiles=[10, 50, 90]
    )
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
        data=[24.0, 24.5, 25.0, 25.5, 26.0], name=UnivariateSimulator.F_PARAMETER_VALUE
    )
    df_test = pd.DataFrame(data=dict_data, index=index)
    assert_frame_equal(aggregated_results.head(), df_test)
