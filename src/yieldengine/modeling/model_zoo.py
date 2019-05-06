from typing import *

from sklearn.base import BaseEstimator


class Model(NamedTuple):
    name: str
    estimator: BaseEstimator
    parameters: Dict[str, Any]


class ModelZoo:
    __slots__ = ["__models"]

    def __init__(self) -> None:
        self.__models = list()

    def add_model(
        self, name: str, estimator: BaseEstimator, parameters: Dict[str, Any]
    ) -> "ModelZoo":
        m = Model(name=name, estimator=estimator, parameters=parameters)
        self.__models.append(m)
        return self

    @property
    def models(self) -> List[Model]:
        return self.__models
