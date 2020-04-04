"""
Core implementation of :mod:`gamma.ml.crossfit`
"""
import logging
from abc import ABCMeta
from copy import copy
from typing import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)

log = logging.getLogger(__name__)

__all__ = ["LearnerCrossfit", "Scoring"]

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)

_INDEX_SENTINEL = pd.Index([])


class Scoring:
    """"
    Basic statistics on the scoring across all cross validation splits of a pipeline.

    :param split_scores: scores of all cross validation splits for a pipeline
    """

    def __init__(self, split_scores: Sequence[float]):
        self._split_scores = np.array(split_scores)
        assert self._split_scores.dtype == float

    def __getitem__(self, item: Union[int, slice]) -> Union[float, np.ndarray]:
        return self._split_scores[item]

    def mean(self) -> float:
        """:return: mean of the split scores"""
        return self._split_scores.mean()

    def std(self) -> float:
        """:return: standard deviation of the split scores"""
        return self._split_scores.std()


class LearnerCrossfit(
    FittableMixin[Sample],
    ParallelizableMixin,
    Generic[T_LearnerPipelineDF],
    metaclass=ABCMeta,
):
    """
    Fits a learner pipeline to all train splits of a given cross-validation strategy,
    and with optional feature shuffling.

    Feature shuffling is active by default, so that every model is trained on a random
    permutation of the feature columns to avoid favouring one of several similar
    features based on column sequence.
    """

    __slots__ = [
        "pipeline",
        "cv",
        "n_jobs",
        "shared_memory",
        "verbose",
        "_model_by_split",
    ]

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        cv: BaseCrossValidator,
        *,
        shuffle_features: Optional[bool] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param pipeline: learner pipeline to be fitted
        :param cv: the cross validator generating the train splits
        :param shuffle_features: if `True`, shuffle column order of features for every \
            crossfit (default: `False`)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.pipeline = pipeline.clone()  #: the learner pipeline being trained
        self.cv = cv  #: the cross validator
        self.shuffle_features: bool = (
            False if shuffle_features is None else shuffle_features
        )
        self.random_state = random_state

        self._model_by_split: Optional[List[T_LearnerPipelineDF]] = None
        self._training_sample: Optional[Sample] = None

    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    def fit(self: T_Self, sample: Sample, **fit_params) -> T_Self:
        """
        Fit the base estimator to the full sample, and fit a clone of the base
        estimator to each of the train splits generated by the cross-validator
        :param sample: the sample to fit the estimators to
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the base estimator
        :return: `self`
        """

        self: LearnerCrossfit  # support type hinting in PyCharm

        pipeline = self.pipeline

        features: pd.DataFrame = sample.features
        target = sample.target
        pipeline.fit(X=features, y=target, **fit_params)

        # we get the features that the learner receives after the preprocessing step
        learner_features = pipeline.features_out
        n_learner_features = len(learner_features)

        feature_sequence_iter: Iterator[Tuple[Optional[pd.Index]]]

        if self.shuffle_features:
            # we are shuffling features, so we create an infinite iterator
            # that creates a new random permutation of feature indices on each
            # iteration
            random_state = check_random_state(self.random_state)
            # noinspection PyTypeChecker
            feature_sequence_iter = iter(
                lambda: (
                    learner_features[random_state.permutation(n_learner_features)],
                ),
                None,
            )
        else:
            # we are not shuffling features, hence we create an infinite iterator of
            # always the same slice that preserves the existing feature sequence
            # noinspection PyTypeChecker
            feature_sequence_iter = iter(lambda: (None,), None)

        with self._parallel() as parallel:
            model_by_split: List[T_LearnerPipelineDF] = parallel(
                self._delayed(LearnerCrossfit._fit_model_for_split)(
                    pipeline.clone(),
                    features.iloc[train_indices],
                    target.iloc[train_indices],
                    feature_sequence,
                    **fit_params,
                )
                for (feature_sequence,), (train_indices, _) in zip(
                    feature_sequence_iter, self.cv.split(features, target)
                )
            )

        self._model_by_split = model_by_split
        self._training_sample = sample

        return self

    def resize(self: T_Self, n_splits: int) -> T_Self:
        """
        Reduce the size of this crossfit by removing a subset of the fits.
        :param n_splits: the number of fits to keep. Must be lower than the number of
            fits
        :return:
        """
        self: LearnerCrossfit

        # ensure that arg n_split has a valid value
        if n_splits > self.get_n_splits():
            raise ValueError(
                f"arg n_splits={n_splits} must not be greater than the number of splits"
                f"in the original crossfit ({self.get_n_splits()} splits)"
            )
        elif n_splits < 1:
            raise ValueError(f"arg n_splits={n_splits} must be a positive integer")

        # copy self and only keep the specified number of fits
        new_crossfit = copy(self)
        new_crossfit._model_by_split = self._model_by_split[:n_splits]
        return new_crossfit

    @property
    def is_fitted(self) -> bool:
        """`True` if the delegate estimator is fitted, else `False`"""
        return self._training_sample is not None

    def get_n_splits(self) -> int:
        """
        Number of splits used for this crossfit.
        """
        self._ensure_fitted()
        return len(self._model_by_split)

    def splits(self) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        """
        :return: an iterator of all train/test splits used by this crossfit
        """
        self._ensure_fitted()

        # ensure we do not return more splits than we have fitted models
        # this is relevant if this is a resized learner crossfit
        return (
            s
            for s, _ in zip(
                self.cv.split(
                    X=self._training_sample.features, y=self._training_sample.target
                ),
                self._model_by_split,
            )
        )

    def models(self) -> Iterator[T_LearnerPipelineDF]:
        """Iterator of all models fitted on the cross-validation train splits."""
        self._ensure_fitted()
        return iter(self._model_by_split)

    @property
    def training_sample(self) -> Sample:
        """The sample used to train this crossfit."""
        self._ensure_fitted()
        return self._training_sample

    # noinspection PyPep8Naming
    @staticmethod
    def _fit_model_for_split(
        pipeline: T_LearnerPipelineDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        feature_sequence: pd.Index,
        **fit_params,
    ) -> T_LearnerPipelineDF:
        return pipeline.fit(X=X, y=y, feature_sequence=feature_sequence, **fit_params)
