# coding=utf-8
"""This module contains classes for:
 - feature selection
 - model creation
 - cross validation
 - model inspection
 """
from typing import *

from sklearn import clone
from sklearn.base import BaseEstimator

from yieldengine.df.pipeline import PipelineDF
from yieldengine.df.transform import DataFrameTransformer


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
        """The underlying `PipelineDF`.
        
        It has two steps: `preprocessing` and `estimator`.
        """
        return PipelineDF(
            steps=[
                (Model.STEP_PREPROCESSING, self.preprocessing),
                (Model.STEP_ESTIMATOR, self.estimator),
            ]
        )

    @property
    def preprocessing(self) -> DataFrameTransformer:
        """The `DataFrameTransformer` used as preprocessing step in the pipeline."""
        return self._preprocessing

    @property
    def estimator(self) -> BaseEstimator:
        """The `DataFrameTransformer` used as estimator in the pipeline."""
        return self._estimator

    def clone(self, parameters: Optional[Dict[str, Any]] = None) -> "Model":
        """Returns a clone Model.

        :param parameters: parameters used to reset the parameters of the pipeline
        :return: a cloned `Model`
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
            new_model.pipeline().set_params(**parameters)

        return new_model
