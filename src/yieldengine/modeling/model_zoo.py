from collections import namedtuple
from sklearn.base import BaseEstimator
from typing import Dict, List

Model = namedtuple("Model", "name estimator parameters")


class ModelZoo:
    def __init__(self):
        self.__models = []

    def add_model(
        self, name: str, estimator: BaseEstimator, parameters: Dict
    ) -> "ModelZoo":
        m = Model(name=name, estimator=estimator, parameters=parameters)
        self.__models.append(m)
        return self

    @property
    def models(self) -> List[Model]:
        return self.__models
