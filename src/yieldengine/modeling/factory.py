from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing import *


class ModelPipelineFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        raise NotImplementedError("call to abstract make_pipeline()")

    @abstractmethod
    def make_multi_estimator_pipeline(
        self, estimators: Iterable[BaseEstimator]
    ) -> Pipeline:
        raise NotImplementedError("call to abstract make_pipeline()")


class PreprocessedModelPipelineFactory(ModelPipelineFactory):
    def __init__(self, preprocessing: Pipeline):
        super().__init__()
        self._preprocessing = preprocessing

    def make_pipeline(self, estimator: BaseEstimator) -> Pipeline:
        pass

    def make_multi_estimator_pipeline(
        self, estimators: Iterable[BaseEstimator]
    ) -> Pipeline:
        pass
