"""
Models to make predictions.

:class:`Model` specifies a model as a pipeline with two steps:
- a preprocessing step
- an estimator step
"""

from typing import *

import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from yieldengine.df.predict import DataFrameClassifierWrapper
from yieldengine.df.transform import DataFrameTransformerWrapper, df_estimator

__all__ = ["Model"]

# todo: replace this with DataFramePredictorWrapper once we have provided DF versions of all
#       sklearn regressors and classifiers
Predictor = TypeVar(
    "Predictor", bound=Union[RegressorMixin, ClassifierMixin] and BaseEstimator
)


# todo: rename to PredictiveWorkflow (tbc)
@df_estimator(df_estimator_type=DataFrameClassifierWrapper)
class Model(BaseEstimator, Generic[Predictor]):
    """
    Specify the preprocessing step and the estimator for a model.

    A model creates a pipeline for a preprocessing transformer (optional; possibly a
    pipeline itself) and an estimator.

    :param predictor: the base estimator used in the pipeline
    :type predictor: :class:`.DataFramePredictor`
    :param preprocessing: the preprocessing step in the pipeline (None or \
      `DataFrameTransformer`)
    """

    def __init__(
        self,
        predictor: Predictor,
        preprocessing: Optional[DataFrameTransformerWrapper] = None,
    ) -> None:
        super().__init__()

        if preprocessing is not None and not isinstance(
            preprocessing, DataFrameTransformerWrapper
        ):
            raise ValueError(
                "arg preprocessing expected to be a " "DataFrameTransformerWrapper"
            )

        self.preprocessing = preprocessing
        self.predictor = predictor

    @property
    def pipeline(self) -> "Model":
        return self

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "Model[Predictor]":
        self.predictor.fit(self._pre_fit_transform(X, y, **fit_params), y, **fit_params)
        return self

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        return self.predictor.predict(self._pre_transform(X), **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        return self.predictor.fit_predict(
            self._pre_fit_transform(X, y, **fit_params), y, **fit_params
        )

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        return self.predictor.predict_proba(self._pre_transform(X))

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        return self.predictor.predict_log_proba(self._pre_transform(X))

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return self.predictor.decision_function(self._pre_transform(X))

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        if sample_weight is None:
            return self.predictor.score(self._pre_transform(X), y)
        else:
            return self.predictor.score(
                self._pre_transform(X), y, sample_weight=sample_weight
            )

    # noinspection PyPep8Naming
    def _pre_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessing is not None:
            return self.preprocessing.transform(X)
        else:
            return X

    # noinspection PyPep8Naming
    def _pre_fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ) -> pd.DataFrame:
        if self.preprocessing is not None:
            return self.preprocessing.fit_transform(X, y, **fit_params)
        else:
            return X

    def clone(self, parameters: Optional[Dict[str, Any]] = None) -> "Model":
        """
        Create an unfitted clone this model with new parameters.

        :param parameters: parameters to set in the cloned the model (optional)
        :return: the cloned `Model`
        """

        copy = clone(self)

        if parameters is not None:
            copy.set_params(**parameters)

        return copy
