import hashlib
import logging
import warnings

from gamma.ml import Sample
from gamma.ml.selection import ClassifierRanker, ParameterGrid
from gamma.ml.validation import BootstrapCV
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF

log = logging.getLogger(__name__)


N_SPLITS = 10


def test_prediction_classifier(n_jobs, iris_sample: Sample) -> None:
    checksum_summary_report = "582bb7e9152c858438a272b096745637"

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    test_cv = BootstrapCV(n_splits=N_SPLITS, random_state=42)

    # define parameters and crossfit
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(), preprocessing=None
            ),
            learner_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    test_sample: Sample = iris_sample

    model_ranker = ClassifierRanker(
        grid=models, cv=test_cv, scoring="f1_macro", n_jobs=n_jobs
    ).fit(sample=test_sample)

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    assert (
        hashlib.md5(model_ranker.summary_report().encode("utf-8")).hexdigest()
    ) == checksum_summary_report

    # consider: model_with_type(...) function for ModelRanking
    crossfit = model_ranker.best_model_crossfit

    assert crossfit.is_fitted
