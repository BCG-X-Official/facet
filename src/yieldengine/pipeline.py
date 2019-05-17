from abc import ABC
from sklearn.pipeline import Pipeline
from yieldengine.feature.transform import (
    DataFrameTransformer,
    SimpleImputerDF,
    OneHotEncoderDF,
)
from yieldengine.model.selection import Model
from typing import *
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class TransformationStep(NamedTuple):
    name: str
    transformer: DataFrameTransformer
    columns: Sequence[str]

    def column_transformer_step(
        self
    ) -> Tuple[str, DataFrameTransformer, Sequence[str]]:
        return (self.name, self.transformer, self.columns)


class ModelPipeline(ABC):
    """
    Abstract yield-engine model pipeline
    """

    __slots__ = ["_estimator", "_pipeline"]

    STEP_PREPROCESSING = "preprocess"
    STEP_MODEL = "model"

    def __init__(
        self, transformations: Sequence[TransformationStep], estimator: BaseEstimator
    ) -> None:
        self._estimator = estimator
        self._pipeline = self._make_pipeline(
            transformations=transformations, estimator=estimator
        )

    def _make_pipeline(
        self, transformations: Sequence[TransformationStep], estimator: BaseEstimator
    ) -> Pipeline:

        column_transformer = ColumnTransformer(
            [step.column_transformer_step() for step in transformations]
        )

        return Pipeline(
            [
                (ModelPipeline.STEP_PREPROCESSING, column_transformer),
                ModelPipeline.STEP_MODEL,
                estimator,
            ]
        )

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator


class SimpleModelPipeline(ModelPipeline):
    STEP_IMPUTE = "impute"
    STEP_ONE_HOT_ENCODE = "one-hot-encode"

    def __init__(
        self,
        model: Model,
        impute_mean: Sequence[str] = None,
        one_hot_encode: Sequence[str] = None,
    ) -> None:

        transformers = []

        if impute_mean is not None and len(impute_mean) > 0:
            imputer = SimpleImputer(strategy="mean")
            transformers.append(
                TransformationStep(
                    name=SimpleModelPipeline.STEP_IMPUTE,
                    transformer=SimpleImputerDF(imputer=imputer),
                    columns=impute_mean,
                )
            )

        if one_hot_encode is not None and sum(1 for col in one_hot_encode) > 0:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            transformers.append(
                TransformationStep(
                    name=SimpleModelPipeline.STEP_ONE_HOT_ENCODE,
                    transformer=OneHotEncoderDF(encoder=encoder),
                    columns=one_hot_encode,
                )
            )

        super().__init__(transformations=transformers, estimator=model.estimator)
