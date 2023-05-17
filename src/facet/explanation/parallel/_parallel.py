"""
Running multiple explainers in parallel using a :class:`ParallelExplainer` instance.
"""

import logging
from multiprocessing.synchronize import Lock as LockType
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from shap import Explainer, Explanation

from pytools.api import AllTracker, inheritdoc
from pytools.parallelization import Job, JobQueue, JobRunner, ParallelizableMixin

from .._types import ArraysAny, ArraysFloat, CatboostPool, XType, YType
from ..base import BaseExplainer

log = logging.getLogger(__name__)

__all__ = [
    "ExplainerJob",
    "ExplainerQueue",
    "ParallelExplainer",
]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


@inheritdoc(match="""[see superclass]""")
class ExplainerJob(Job[ArraysAny]):
    """
    A call to an explanation function with given `X` and `y` values.
    """

    #: the SHAP explainer to use
    explainer: BaseExplainer

    #: if ``False``, calculate SHAp values; otherwise, calculate SHAP interaction values
    interactions: bool

    #: the feature values of the observations to be explained
    X: XType

    #: the target values of the observations to be explained
    y: YType

    #: additional arguments specific to the explanation method
    kwargs: Dict[str, Any]

    # noinspection PyPep8Naming
    def __init__(
        self,
        explainer: BaseExplainer,
        X: XType,
        y: YType = None,
        *,
        interactions: bool,
        **kwargs: Any,
    ) -> None:
        """
        :param explainer: the SHAP explainer to use
        :param X: the feature values of the observations to be explained
        :param y: the target values of the observations to be explained
        :param interactions: if ``False``, calculate SHAP values; if ``True``,
            calculate SHAP interaction values
        :param kwargs: additional arguments specific to the explanation method
        """
        self.explainer = explainer
        self.X = X
        self.y = y
        self.interactions = interactions
        self.kwargs = kwargs

    def run(self) -> ArraysAny:
        """[see superclass]"""

        shap_fn: Callable[..., ArraysAny] = (
            self.explainer.shap_interaction_values
            if self.interactions
            else self.explainer.shap_values
        )

        if self.y is None:
            return shap_fn(self.X, **self.kwargs)
        else:
            return shap_fn(self.X, self.y, **self.kwargs)


@inheritdoc(match="""[see superclass]""")
class ExplainerQueue(JobQueue[ArraysAny, ArraysAny]):
    """
    A queue splitting a data set to be explained into multiple jobs.
    """

    # defined in superclass, repeated here for Sphinx
    lock: LockType

    #: the SHAP explainer to use
    explainer: BaseExplainer

    #: if ``False``, calculate SHAP values; otherwise, calculate SHAP interaction values
    interactions: bool

    #: the feature values of the observations to be explained
    X: npt.NDArray[Any]

    #: the target values of the observations to be explained
    y: Optional[npt.NDArray[Any]]

    #: the maximum number of observations to allocate to each job
    max_job_size: int

    #: additional arguments specific to the explanation method
    kwargs: Dict[str, Any]

    # noinspection PyPep8Naming
    def __init__(
        self,
        explainer: BaseExplainer,
        X: XType,
        y: YType = None,
        *,
        interactions: bool,
        max_job_size: int,
        **kwargs: Any,
    ) -> None:
        """
        :param explainer: the SHAP explainer to use
        :param X: the feature values of the observations to be explained
        :param y: the target values of the observations to be explained
        :param interactions: if ``False``, calculate SHAP values; if ``True``,
            calculate SHAP interaction values
        :param max_job_size: the maximum number of observations to allocate to each job
        :param kwargs: additional arguments specific to the explanation method

        :raise NotImplementedError: if `X` is a :class:`~catboost.Pool`;
            this is currently not supported
        """
        super().__init__()

        if isinstance(X, CatboostPool):
            raise NotImplementedError("CatBoost Pool is not supported")
        self.explainer = explainer
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y
        self.interactions = interactions
        self.max_job_size = max_job_size
        self.kwargs = kwargs

    def jobs(self) -> Iterable[Job[ArraysAny]]:
        """[see superclass]"""

        x = self.X
        y = self.y
        n = len(x)
        job_size = (n - 1) // len(self) + 1
        kwargs = self.kwargs

        return (
            ExplainerJob(
                self.explainer,
                X=x[start : start + job_size].copy(),
                y=None if y is None else y[start : start + job_size].copy(),
                interactions=self.interactions,
                **kwargs,
            )
            for start in range(0, n, job_size)
        )

    def aggregate(self, job_results: List[ArraysAny]) -> ArraysAny:
        """[see superclass]"""
        if isinstance(job_results[0], np.ndarray):
            return np.vstack(job_results)
        else:
            return [np.vstack(arrays) for arrays in zip(*job_results)]

    def __len__(self) -> int:
        return (len(self.X) - 1) // self.max_job_size + 1


@inheritdoc(match="""[see superclass]""")
class ParallelExplainer(BaseExplainer, ParallelizableMixin):
    """
    A wrapper class, turning an explainer into a parallelized version, explaining
    chunks of observations in parallel.
    """

    #: The explainer being parallelized by this wrapper
    explainer: BaseExplainer

    #: the maximum number of observations to allocate to any of the explanation jobs
    #: running in parallel
    max_job_size: int

    # defined in superclass, repeated here for Sphinx:
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

    def __init__(
        self,
        explainer: BaseExplainer,
        *,
        max_job_size: int = 10,
        n_jobs: Optional[int],
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param explainer: the explainer to be parallelized by this wrapper
        :param max_job_size: the maximum number of observations to allocate to any of
            the explanation jobs running in parallel
        """
        Explainer.__init__(self, model=explainer.model)
        ParallelizableMixin.__init__(
            self,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        if isinstance(explainer, ParallelExplainer):
            log.warning(
                f"creating parallel explainer from parallel explainer {explainer!r}"
            )

        self.explainer = explainer
        self.max_job_size = max_job_size

    assert __init__.__doc__ is not None
    __init__.__doc__ += cast(str, ParallelizableMixin.__init__.__doc__)

    @property
    def supports_interaction(self) -> bool:
        """[see superclass]"""
        return self.explainer.supports_interaction

    def __call__(self, *args: Any, **kwargs: Any) -> Explanation:
        """
        Forward the call to the wrapped explainer.

        :param args: positional arguments to be forwarded to the wrapped explainer
        :param kwargs: keyword arguments to be forwarded to the wrapped explainer
        :return: the explanation returned by the wrapped explainer
        """
        return self.explainer(*args, **kwargs)

    # noinspection PyPep8Naming
    def shap_values(self, X: XType, y: YType = None, **kwargs: Any) -> ArraysFloat:
        """[see superclass]"""
        if y is None:
            return self.explainer.shap_values(X=X, **kwargs)
        else:
            return self.explainer.shap_values(X=X, y=y, **kwargs)

    # noinspection PyPep8Naming
    def shap_interaction_values(
        self, X: XType, y: YType = None, **kwargs: Any
    ) -> ArraysFloat:
        """[see superclass]"""
        if y is None:
            return self.explainer.shap_interaction_values(X=X, **kwargs)
        else:
            return self.explainer.shap_interaction_values(X=X, y=y, **kwargs)

    # noinspection PyPep8Naming
    def _run(
        self,
        explainer: BaseExplainer,
        X: XType,
        y: YType = None,
        *,
        interactions: bool,
        **kwargs: Any,
    ) -> ArraysFloat:
        return JobRunner.from_parallelizable(self).run_queue(
            ExplainerQueue(
                explainer=explainer,
                X=X,
                y=y,
                interactions=interactions,
                max_job_size=self.max_job_size,
                **kwargs,
            )
        )
