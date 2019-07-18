from typing import *

from sklearn import clone
from sklearn.base import ClassifierMixin, RegressorMixin

from yieldengine.df.pipeline import PipelineDF
from yieldengine.df.predict import DataFramePredictor
from yieldengine.df.transform import DataFrameTransformer

__all__ = ["Model"]

# todo: replace this with DataFramePredictor once we have provided DF versions of all
#       sklearn regressors and classifiers
Predictor = Union[DataFramePredictor, RegressorMixin, ClassifierMixin]


# todo: rename to PredictiveWorkflow (tbc)
class Model:
    """
    A model can create a pipeline for a preprocessing transformer (optional; possibly a
    pipeline itself) and an estimator.

    :param Estimator predictor: the base estimator used in the pipeline
    :param preprocessing: the preprocessing step in the pipeline (None or \
    `DataFrameTransformer`)
    """

    __slots__ = ["_pipeline", "_preprocessing", "_predictor"]

    STEP_PREPROCESSING = "preprocessing"
    STEP_ESTIMATOR = "estimator"

    def __init__(
        self, predictor: Predictor, preprocessing: Optional[DataFrameTransformer] = None
    ) -> None:
        super().__init__()

        if preprocessing is not None and not isinstance(
            preprocessing, DataFrameTransformer
        ):
            raise ValueError(
                "arg preprocessing expected to be a " "DataFrameTransformer"
            )

        self._predictor = predictor
        self._preprocessing = preprocessing
        self._pipeline = PipelineDF(
            steps=[
                (Model.STEP_PREPROCESSING, self.preprocessing),
                (Model.STEP_ESTIMATOR, self.predictor),
            ]
        )

    @property
    def pipeline(self) -> PipelineDF:
        return self._pipeline

    @property
    def preprocessing(self) -> DataFrameTransformer:
        return self._preprocessing

    @property
    def predictor(self) -> Predictor:
        return self._predictor

    def clone(self, parameters: Optional[Dict[str, Any]] = None) -> "Model":
        """
         Clone this model.

        :param parameters: parameters used to reset the model parameters
        :return: the cloned model
        """
        predictor = clone(self._predictor)
        preprocessing = self._preprocessing
        if preprocessing is not None:
            preprocessing = clone(preprocessing)

        my_class: Type[Model] = self.__class__
        new_model: Model = my_class(predictor=predictor, preprocessing=preprocessing)

        # to set the parameters, we need to wrap the preprocessor and estimator in a
        # pipeline object
        if parameters is not None:
            new_model.pipeline.set_params(**parameters)

        return new_model
