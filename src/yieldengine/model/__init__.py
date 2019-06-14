from typing import *

from sklearn import clone
from sklearn.base import BaseEstimator

from yieldengine.df.transform import DataFrameTransformer
from yieldengine.model.pipeline import PipelineDF


class Model:
    """
    A model can create a pipeline for a preprocessing transformer (optional; possibly a
    pipeline itself) and an estimator
    """

    __slots__ = ["_pipeline", "_preprocessing", "_estimator"]

    STEP_PREPROCESSING = "preprocessing"
    STEP_ESTIMATOR = "estimator"

    def __init__(
        self,
        estimator: BaseEstimator,
        preprocessing: Optional[DataFrameTransformer] = None,
    ) -> None:
        super().__init__()

        if preprocessing is not None and not isinstance(
            preprocessing, DataFrameTransformer
        ):
            raise ValueError(
                "arg preprocessing expected to be a " "DataFrameTransformer"
            )

        self._estimator = estimator
        self._preprocessing = preprocessing

    def pipeline(self) -> PipelineDF:
        return PipelineDF(
            steps=[
                (Model.STEP_PREPROCESSING, self.preprocessing),
                (Model.STEP_ESTIMATOR, self.estimator),
            ]
        )

    @property
    def preprocessing(self) -> DataFrameTransformer:
        return self._preprocessing

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    def clone(self, parameters: Optional[Dict[str, Any]] = None) -> "Model":
        estimator = clone(self._estimator)
        preprocessing = self._preprocessing
        if preprocessing is not None:
            preprocessing = clone(preprocessing)

        new_model = self.__class__(estimator=estimator, preprocessing=preprocessing)

        # to set the parameters, we need to wrap the preprocessor and estimator in a
        # pipeline object
        if parameters is not None:
            new_model.pipeline().set_params(**parameters)

        return new_model
