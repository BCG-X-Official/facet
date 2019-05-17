from typing import *

from sklearn import clone
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


class ModelPipeline(Pipeline):
    """
    Yield-engine model pipeline
    """

    __slots__ = ["_preprocessing"]

    STEP_MODEL = "model"
    STEP_PREPROCESSING = "preprocessor"

    def __init__(
        self,
        preprocessing: Union[Iterable[TransformationStep], None],
        estimator: BaseEstimator,
    ) -> None:

        if preprocessing is None or isinstance(preprocessing, TransformationStep):
            raise ValueError("expected an Iterable[TransformationStep]")

        self._preprocessing = preprocessing

        preprocessing = list(preprocessing)

        super().__init__(
            [
                (ModelPipeline.STEP_PREPROCESSING, Pipeline(preprocessing)),
                (ModelPipeline.STEP_MODEL, estimator),
            ]
        )

    @property
    def preprocessing_pipeline(self) -> Pipeline:
        return self.named_steps[ModelPipeline.STEP_PREPROCESSING]

    @property
    def last_preprocessing(self) -> DataFrameTransformer:
        return self.preprocessing_pipeline.steps[-1][1]

    @property
    def estimator(self) -> BaseEstimator:
        return self.named_steps[ModelPipeline.STEP_MODEL]

    def copy(self) -> "ModelPipeline":

        preprocessing = None

        if self._preprocessing is not None:
            if isinstance(self._preprocessing, ColumnTransformerDF):
                preprocessing = clone(self._preprocessing)
            else:
                preprocessing = [
                    TransformationStep(name=t.name, transformer=clone(t.transformer))
                    for t in self._preprocessing
                ]

        return ModelPipeline(
            preprocessing=preprocessing, estimator=clone(self.estimator)
        )

    def has_transformations(self) -> bool:
        return ModelPipeline.STEP_PREPROCESSING in self.named_steps


STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


def make_simple_transformer_step(
    impute_mean: Sequence[str] = None, one_hot_encode: Sequence[str] = None
) -> TransformationStep:

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

    return TransformationStep(
        name="t1", transformer=ColumnTransformerDF(transformers=column_transforms)
    )
