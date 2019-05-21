import collections
from typing import *

from sklearn.base import BaseEstimator, clone

from yieldengine.feature.transform import (
    ColumnTransformerDF,
    DataFrameTransformer,
    OneHotEncoderDF,
    SimpleImputerDF,
)
from yieldengine.model.pipeline import PipelineDF

TransformationStep = Tuple[str, DataFrameTransformer]


class ModelPipeline:
    """
    A model configuration can make a pipeline for a specif
    """

    __slots__ = ["_pipeline"]

    STEP_MODEL = "estimator"

    def __init__(
        self,
        preprocessing: Optional[Sequence[TransformationStep]],
        estimator: BaseEstimator,
    ) -> None:
        super().__init__()

        if preprocessing is None:
            preprocessing = []
        elif isinstance(preprocessing, collections.abc.Sequence):
            for step in preprocessing:
                if not (
                    isinstance(step, tuple)
                    and len(step) == 2
                    and isinstance(step[0], str)
                    and isinstance(step[1], DataFrameTransformer)
                ):
                    raise ValueError(
                        "arg preprocessing must only contain instances of "
                        "[str, DataFrameTransformer] pairs"
                    )
        else:
            raise ValueError(
                "arg preprocessing must be a sequence of make_pipeline transformation steps"
            )

        self._pipeline = PipelineDF(
            steps=[*preprocessing, (ModelPipeline.STEP_MODEL, estimator)]
        )

    @property
    def pipeline(self) -> PipelineDF:
        return self._pipeline

    def make_pipeline(self) -> PipelineDF:
        return clone(self._pipeline)

    @property
    def estimator(self) -> BaseEstimator:
        return self._pipeline.steps[-1][1]


STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


def make_simple_transformer_step(
    impute_median: Sequence[str] = None, one_hot_encode: Sequence[str] = None
) -> Tuple[str, DataFrameTransformer]:

    column_transforms = []

    if impute_median is not None and len(impute_median) > 0:
        column_transforms.append(
            (STEP_IMPUTE, SimpleImputerDF(strategy="median"), impute_median)
        )

    if one_hot_encode is not None and len(one_hot_encode) > 0:
        column_transforms.append(
            (
                STEP_ONE_HOT_ENCODE,
                OneHotEncoderDF(sparse=False, handle_unknown="ignore"),
                one_hot_encode,
            )
        )

    return "t1", ColumnTransformerDF(transformers=column_transforms)
