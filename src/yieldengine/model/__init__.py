"""
This package defines a :class:`~Model` model class and other classes to inspect the
model and add cross validation support.
"""

from typing import *

from sklearn import clone
from sklearn.base import BaseEstimator

from yieldengine.df.pipeline import PipelineDF
from yieldengine.df.transform import DataFrameTransformer


class Model:
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
        """The underlying pipeline of the model.

        It has two steps: ``preprocessing`` and ``estimator``."""
        return self._pipeline

    @property
    def preprocessing(self) -> DataFrameTransformer:
        """The ``preprocessing`` step of the pipeline."""
        return self._preprocessing

    @property
    def estimator(self) -> BaseEstimator:
        """The ``estimator`` step of the pipeline."""
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
