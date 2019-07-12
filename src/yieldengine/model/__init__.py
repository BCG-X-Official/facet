from abc import ABC
from enum import Enum
from typing import *

from sklearn import clone
from sklearn.base import BaseEstimator

from yieldengine.df.pipeline import PipelineDF
from yieldengine.df.transform import DataFrameTransformer


class Model(ABC):
    """
    A model can create a pipeline for a preprocessing transformer (optional; possibly a
    pipeline itself) and an estimator.

    :param BaseEstimator estimator: the base estimator used in the pipeline
    :param preprocessing: the preprocessing step in the pipeline (None or \
    `DataFrameTransformer`)
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
        self._pipeline = PipelineDF(
            steps=[
                (Model.STEP_PREPROCESSING, self.preprocessing),
                (Model.STEP_ESTIMATOR, self.estimator),
            ]
        )

    @property
    def pipeline(self) -> PipelineDF:
        return self._pipeline

    @property
    def preprocessing(self) -> DataFrameTransformer:
        return self._preprocessing

    @property
    def estimator(self) -> BaseEstimator:
        return self._estimator

    def clone(self, parameters: Optional[Dict[str, Any]] = None) -> "Model":
        """
         Clone this model.

        :param parameters: parameters used to reset the model parameters
        :return: the cloned model
        """
        estimator = clone(self._estimator)
        preprocessing = self._preprocessing
        if preprocessing is not None:
            preprocessing = clone(preprocessing)

        my_class: Type[Model] = self.__class__
        new_model: Model = my_class(estimator=estimator, preprocessing=preprocessing)

        # to set the parameters, we need to wrap the preprocessor and estimator in a
        # pipeline object
        if parameters is not None:
            new_model.pipeline.set_params(**parameters)

        return new_model


class RegressionModel(Model):
    pass


class ProbabilityCalibrationMethod(Enum):
    SIGMOID = "sigmoid"
    ISOTONIC = "isotonic"


class ClassificationModel(Model):
    __slots__ = ["_pipeline", "_preprocessing", "_estimator", "_calibration_method"]

    def __init__(
        self,
        estimator: BaseEstimator,
        preprocessing: Optional[DataFrameTransformer] = None,
        calibration_method: Optional[
            ProbabilityCalibrationMethod
        ] = ProbabilityCalibrationMethod.SIGMOID,
    ):
        super().__init__(estimator, preprocessing)
        self._calibration_method = calibration_method

    @property
    def calibration_method(self) -> ProbabilityCalibrationMethod:
        return self.calibration_method
