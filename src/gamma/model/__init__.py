"""
Model pipeline enabled for data frames.

A :class:`ModelPipelineDF` is a special case of a pipeline with two steps:
- a preprocessing step
- an estimator step
"""

from typing import *

import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator

from gamma import ListLike
from gamma.sklearndf import (
    DataFrameClassifier,
    DataFramePredictor,
    DataFrameRegressor,
    DataFrameTransformer,
)

__all__ = ["ModelPipelineDF"]

T_PredictorDF = TypeVar(
    "PredictorDF", bound=Union[DataFrameRegressor, DataFrameClassifier]
)


class ModelPipelineDF(
    BaseEstimator, DataFrameRegressor, DataFrameClassifier, Generic[T_PredictorDF]
):
    """
    A data frame enabled model pipeline with an optional preprocessing step and a
    mandatory estimator step.

    :param preprocessing: the preprocessing step in the pipeline (defaults to ``None``)
    :param predictor: the base estimator used in the pipeline
    :type predictor: :class:`.DataFramePredictor`
    """

    def __init__(
        self,
        predictor: T_PredictorDF,
        preprocessing: Optional[DataFrameTransformer] = None,
    ) -> None:
        super().__init__()

        if preprocessing is not None and not isinstance(
            preprocessing, DataFrameTransformer
        ):
            raise TypeError(
                "arg preprocessing expected to be a DataFrameTransformer but is a "
                f"{type(preprocessing).__name__}"
            )
        if not isinstance(predictor, DataFramePredictor):
            raise TypeError(
                "arg predictor expected to be a DataFramePredictor but is a "
                f"{type(predictor).__name__}"
            )

        self.preprocessing = preprocessing
        self.predictor = predictor

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "ModelPipelineDF[T_PredictorDF]":
        self.predictor.fit(self._pre_fit_transform(X, y, **fit_params), y, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.preprocessing.is_fitted and self.predictor.is_fitted

    @property
    def columns_in(self) -> pd.Index:
        if self.preprocessing is not None:
            return self.preprocessing.columns_in
        else:
            return self.predictor.columns_in

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
        return cast(DataFrameClassifier, self.predictor).predict_proba(
            self._pre_transform(X)
        )

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        return cast(DataFrameClassifier, self.predictor).predict_log_proba(
            self._pre_transform(X)
        )

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return cast(DataFrameClassifier, self.predictor).decision_function(
            self._pre_transform(X)
        )

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

    @property
    def classes(self) -> Optional[ListLike[Any]]:
        return cast(DataFrameClassifier, self.predictor).classes

    @property
    def n_outputs(self) -> int:
        return self.predictor.n_outputs

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

    def clone(self, parameters: Optional[Dict[str, Any]] = None) -> "ModelPipelineDF":
        """
        Create an unfitted clone this model with new parameters.

        :param parameters: parameters to set in the cloned the model (optional)
        :return: the cloned `ModelPipelineDF`
        """

        copy = clone(self)

        if parameters is not None:
            copy.set_params(**parameters)

        return copy
