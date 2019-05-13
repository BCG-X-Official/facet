import logging
import warnings

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC

from yieldengine.loading.sample import Sample
from yieldengine.modeling.selection import Model, ModelRanker, ModelRanking, ScoredModel
from yieldengine.modeling.validation import CircularCrossValidator

log = logging.getLogger(__name__)


def test_model_ranker(
    batch_table: pd.DataFrame, regressor_grids, sample: Sample, preprocessor
) -> None:
    # define the circular cross validator with just 5 folds (to speed up testing)
    circular_cv = CircularCrossValidator(test_ratio=0.20, num_folds=5)

    model_ranker: ModelRanker = ModelRanker(
        models=regressor_grids,
        preprocessing_factory=preprocessor,
        cv=circular_cv,
        scoring="r2",
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: ModelRanking = model_ranker.run(sample=sample)

    assert len(model_ranking) > 0
    assert isinstance(
        model_ranking.model(rank=ModelRanking.BEST_MODEL_RANK), ScoredModel
    )
    assert (
        model_ranking.model(rank=0).ranking_score
        >= model_ranking.model(rank=1).ranking_score
        >= model_ranking.model(rank=2).ranking_score
        >= model_ranking.model(rank=3).ranking_score
        >= model_ranking.model(rank=4).ranking_score
        >= model_ranking.model(rank=len(model_ranking) - 1).ranking_score
    )

    # check if parameters set for estimators actually match expected:
    for r in range(0, len(model_ranking)):
        m: ScoredModel = model_ranking.model(r)
        assert set(m.parameters).issubset(m.estimator.get_params())

    log.info(f"\n{model_ranking}")


def test_model_ranker_no_preprocessing() -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a yield-engine circular CV:
    cv = CircularCrossValidator(test_ratio=0.21, num_folds=50)

    # define parameters and model
    models = [
        Model(
            estimator=SVC(gamma="scale"),
            parameter_grid={"kernel": ("linear", "rbf"), "C": [1, 10]},
        )
    ]

    model_ranker: ModelRanker = ModelRanker(models=models, cv=cv)

    #  load sklearn test-data and convert to pd
    iris = datasets.load_iris()
    test_data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    test_sample: Sample = Sample(observations=test_data, target_name="target")

    model_ranking: ModelRanking = model_ranker.run(test_sample)

    log.info(f"\n{model_ranking.summary_report()}")

    assert (
        model_ranking.model(ModelRanking.BEST_MODEL_RANK).ranking_score >= 0.8
    ), "Expected a performance of at least 0.8"
