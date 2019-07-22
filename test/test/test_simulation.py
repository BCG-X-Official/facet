import logging
from typing import *

import pandas as pd

from gamma import Sample
from gamma.model.prediction import RegressorFitCV
from gamma.model.selection import ModelEvaluation, ModelGrid, ModelRanker
from gamma.model.validation import CircularCrossValidator
from gamma.sklearndf import DataFrameTransformer
from gamma.yieldengine.partition import ContinuousRangePartitioning
from gamma.yieldengine.simulation import UnivariateSimulation

log = logging.getLogger(__name__)

N_SPLITS = 5
TEST_RATIO = 0.2


def test_univariate_simulation(
    batch_table: pd.DataFrame,
    regressor_grids: Iterable[ModelGrid],
    sample: Sample,
    simple_preprocessor: DataFrameTransformer,
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

    sim = UnivariateSimulation(model_fit=model_fit)

    parameterized_feature = "Step4-6 RawMat Vendor Compound08 Purity (#)"

    res = sim.simulate_feature(
        feature_name=parameterized_feature,
        feature_values=ContinuousRangePartitioning(
            values=sample.features.loc[:, parameterized_feature]
        ).partitions(),
    )

    log.debug(res)
    log.debug(
        UnivariateSimulation.aggregate_simulation_results(
            results_per_split=res, percentiles=[10, 50, 90]
        )
    )
