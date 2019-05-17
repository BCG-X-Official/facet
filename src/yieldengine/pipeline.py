from typing import *

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from yieldengine.feature.transform import (
    ColumnTransformerDF,
    DataFrameTransformer,
    OneHotEncoderDF,
    SimpleImputerDF,
)


class TransformationStep(NamedTuple):
    name: str
    transformer: DataFrameTransformer


class ModelPipeline:
    """
    Yield-engine model pipeline
    """

    __slots__ = ["_transformers", "_estimator", "_pipeline", "_last_transformation"]

    STEP_MODEL = "model"

    def __init__(
        self, preprocessing: Iterable[TransformationStep], estimator: BaseEstimator
    ) -> None:
        preprocessing = list(preprocessing)
        self._estimator = estimator
        self._pipeline = Pipeline(
            [preprocessing, (ModelPipeline.STEP_MODEL, estimator)]
        )

        self._last_transformation = (
            preprocessing[-1] if len(preprocessing) > 0 else None
        )

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator


STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


def make_transformation_steps(
    impute_mean: Sequence[str] = None, one_hot_encode: Sequence[str] = None
) -> DataFrameTransformer:

    column_transforms = []

    if impute_mean is not None and len(impute_mean) > 0:
        column_transforms.append(
            (STEP_IMPUTE, SimpleImputerDF(strategy="mean"), impute_mean)
        )

    if one_hot_encode is not None and len(one_hot_encode) > 0:
        column_transforms.append(
            (
                STEP_ONE_HOT_ENCODE,
                OneHotEncoderDF(sparse=False, handle_unknown="ignore"),
                one_hot_encode,
            )
        )

    return ColumnTransformerDF(transformers=column_transforms)
