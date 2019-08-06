"""
Test module for PipelineDF inspired by:
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tests/test_pipeline.py
"""
import shutil
import time
from tempfile import mkdtemp
from typing import *

import joblib
import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.utils.testing import (
    assert_array_equal,
    assert_no_warnings,
    assert_raises,
    assert_raises_regex,
)

from gamma.model import Sample
from gamma.sklearndf import TransformerDF
from gamma.sklearndf._wrapper import df_estimator
from gamma.sklearndf.classification import LogisticRegressionDF, SVCDF
from gamma.sklearndf.pipeline import PipelineDF
from gamma.sklearndf.regression import DummyRegressorDF, LassoDF, LinearRegressionDF
from gamma.sklearndf.transformation import (
    ColumnPreservingTransformerWrapperDF,
    SelectKBestDF,
    SimpleImputerDF,
)


def test_set_params_nested_pipeline_df() -> None:
    """ Test parameter setting for nested pipelines - adapted from
    sklearn.tests.test_pipeline """

    PipelineDF([("b", SimpleImputerDF(strategy="median"))])

    estimator = PipelineDF([("a", PipelineDF([("b", DummyRegressorDF())]))])

    estimator.set_params(a__b__alpha=0.001, a__b=LassoDF())
    estimator.set_params(a__steps=[("b", LogisticRegressionDF())], a__b__C=5)


class NoFit(BaseEstimator, TransformerMixin):
    """Small class to test parameter dispatching.
    """

    def __init__(self, a: str = None, b: str = None) -> None:
        self.a = a
        self.b = b


class NoTrans(NoFit):
    def fit(self, X, y) -> "NoTrans":
        return self

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def set_params(self, **params: Dict[str, Any]) -> "NoTrans":
        self.a = params["a"]
        return self


class NoInvTransf(NoTrans):
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class Transf(NoInvTransf):
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X


class DummyTransf(Transf):
    """Transformer which store the column means"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyTransf":
        self.means_ = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_ = time.time()
        return self


class TransfFitParams(Transf):
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TransfFitParams":
        self.fit_params = fit_params
        return self


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class DummyTransfDF(DummyTransf, TransformerDF):
    """ Wraps a  DummyTransf; accepts and returns data frames """

    pass


@df_estimator(df_wrapper_type=ColumnPreservingTransformerWrapperDF)
class NoTransDF(NoTrans, TransformerDF):
    """ Wraps a  DummyTransf; accepts and returns data frames """

    pass


def test_pipelinedf_memory(iris_sample: Sample) -> None:
    """ Test memory caching in PipelineDF - taken almost 1:1 from
    sklearn.tests.test_pipeline """

    cachedir = mkdtemp()
    try:

        memory = joblib.Memory(location=cachedir, verbose=10)
        # Test with Transformer + SVC
        clf = SVCDF(probability=True, random_state=0)
        transf = DummyTransfDF()
        pipe = PipelineDF([("transf", clone(transf)), ("svc", clf)])
        cached_pipe = PipelineDF([("transf", transf), ("svc", clf)], memory=memory)

        # Memoize the transformer at the first fit
        cached_pipe.fit(iris_sample.features, iris_sample.target)
        pipe.fit(iris_sample.features, iris_sample.target)
        # Get the time stamp of the transformer in the cached pipeline
        ts = cached_pipe.named_steps["transf"].timestamp_
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(
            pipe.predict(iris_sample.features),
            cached_pipe.predict(iris_sample.features),
        )
        assert_array_equal(
            pipe.predict_proba(iris_sample.features),
            cached_pipe.predict_proba(iris_sample.features),
        )
        assert_array_equal(
            pipe.predict_log_proba(iris_sample.features),
            cached_pipe.predict_log_proba(iris_sample.features),
        )
        assert_array_equal(
            pipe.score(iris_sample.features, iris_sample.target),
            cached_pipe.score(iris_sample.features, iris_sample.target),
        )
        assert_array_equal(
            pipe.named_steps["transf"].means_, cached_pipe.named_steps["transf"].means_
        )
        assert not hasattr(transf, "means_")
        # Check that we are reading the cache while fitting
        # a second time
        cached_pipe.fit(iris_sample.features, iris_sample.target)
        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(
            pipe.predict(iris_sample.features),
            cached_pipe.predict(iris_sample.features),
        )
        assert_array_equal(
            pipe.predict_proba(iris_sample.features),
            cached_pipe.predict_proba(iris_sample.features),
        )
        assert_array_equal(
            pipe.predict_log_proba(iris_sample.features),
            cached_pipe.predict_log_proba(iris_sample.features),
        )
        assert_array_equal(
            pipe.score(iris_sample.features, iris_sample.target),
            cached_pipe.score(iris_sample.features, iris_sample.target),
        )
        assert_array_equal(
            pipe.named_steps["transf"].means_, cached_pipe.named_steps["transf"].means_
        )
        assert ts == cached_pipe.named_steps["transf"].timestamp_
        # Create a new pipeline with cloned estimators
        # Check that even changing the name step does not affect the cache hit
        clf_2 = SVCDF(probability=True, random_state=0)
        transf_2 = DummyTransfDF()
        cached_pipe_2 = PipelineDF(
            [("transf_2", transf_2), ("svc", clf_2)], memory=memory
        )
        cached_pipe_2.fit(iris_sample.features, iris_sample.target)

        # Check that cached_pipe and pipe yield identical results
        assert_array_equal(
            pipe.predict(iris_sample.features),
            cached_pipe_2.predict(iris_sample.features),
        )
        assert_array_equal(
            pipe.predict_proba(iris_sample.features),
            cached_pipe_2.predict_proba(iris_sample.features),
        )
        assert_array_equal(
            pipe.predict_log_proba(iris_sample.features),
            cached_pipe_2.predict_log_proba(iris_sample.features),
        )
        assert_array_equal(
            pipe.score(iris_sample.features, iris_sample.target),
            cached_pipe_2.score(iris_sample.features, iris_sample.target),
        )
        assert_array_equal(
            pipe.named_steps["transf"].means_,
            cached_pipe_2.named_steps["transf_2"].means_,
        )
        assert ts == cached_pipe_2.named_steps["transf_2"].timestamp_
    finally:
        shutil.rmtree(cachedir)


def test_pipelinedf__init() -> None:
    """ Test the various init parameters of the pipeline. """

    assert_raises(TypeError, PipelineDF)
    # Check that we can't instantiate pipelines with objects without fit
    # method
    assert_raises_regex(
        TypeError,
        "Last step of Pipeline should implement fit "
        "or be the string 'passthrough'"
        ".*NoFit.*",
        PipelineDF,
        [("clf", NoFit())],
    )

    # Smoke test with only an estimator
    clf = NoTransDF()
    pipe = PipelineDF([("svc", clf)])
    assert pipe.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **pipe.get_params(deep=False)
    )

    # Check that params are set
    pipe.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(pipe)

    # Test with two objects
    clf = SVCDF()
    filter1 = SelectKBestDF(f_classif)
    pipe = PipelineDF([("anova", filter1), ("svc", clf)])

    # Check that estimators are not cloned on pipeline construction
    assert pipe.named_steps["anova"] is filter1
    assert pipe.named_steps["svc"] is clf

    # todo: decide if this assertion is needed - currently can't be tested since
    #  functions such as fit, transform, etc. are provided by sklearndf, even though
    #  delegate_estimator is not guaranteed to have them!
    # Check that we can't instantiate with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    # assert_raises_regex(
    #    TypeError,
    #    "All intermediate steps should be transformers" ".*\\bNoTrans\\b.*",
    #    PipelineDF,
    #    [("t", NoTransDF()), ("svc", clf)],
    # )

    # Check that params are set
    pipe.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(pipe)

    # Check that params are not set when naming them wrong
    assert_raises(ValueError, pipe.set_params, anova__C=0.1)

    # Test clone
    pipe2 = assert_no_warnings(clone, pipe)
    assert not pipe.named_steps["svc"] is pipe2.named_steps["svc"]

    # Check that apart from estimators, the parameters are the same
    params = pipe.get_params(deep=True)
    params2 = pipe2.get_params(deep=True)

    for x in pipe.get_params(deep=False):
        params.pop(x)

    for x in pipe2.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")

    assert params == params2


def test_pipelinedf_raise_set_params_error() -> None:
    """ Test pipeline raises set params error message for nested models. """
    pipe = PipelineDF([("cls", LinearRegressionDF())])

    # expected error message
    error_msg = (
        "Invalid parameter %s for estimator Pipeline. "
        "Check the list of available parameters "
        "with `estimator.get_params().keys()`."
    )

    assert_raises_regex(
        ValueError,
        "Invalid parameter fake for estimator Pipeline",
        pipe.set_params,
        fake="nope",
    )

    # nested model check
    assert_raises_regex(
        ValueError,
        "Invalid parameter fake for estimator Pipeline",
        pipe.set_params,
        fake__estimator="nope",
    )
