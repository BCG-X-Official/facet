from typing import *

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from yieldengine.loading.sample import Sample
from yieldengine.modeling.validation import CircularCrossValidator


class ModelPredictor:
    def __init__(
        self,
        estimator: BaseEstimator,
        cv: CircularCrossValidator,
        preprocessing: Pipeline = None,
    ) -> None:
        self._estimator = estimator
        self._cv = cv
        self._preprocessing = preprocessing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
        pass

    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    def foldwise_fit_predict(
        self, observations: Sample
    ) -> Generator[Tuple[Pipeline, pd.Series]]:
        # 1. wrap self._estimator in a Pipeline if self._preprocessing is not None
        # 2. get fold splits across observations using self._cv
        # 3. for each split
        #       - fit Pipeline on X of train set
        #       - predict y for test set
        #       - yield deepcopy(Pipeline), predictions (!ensure predictions has index!)
        pass


class ModelInspector:
    def __init__(self, predictor: ModelPredictor) -> None:
        self._predictor = predictor
        pass

    def shap_value_matrix(self, observations: Sample) -> pd.DataFrame:
        for estimator, predictions in self._predictor.foldwise_fit_predict(
            observations=observations
        ):
            pass
            # 1.) init the Shap explainer:
            # explainer = shap.TreeExplainer(estimator)

            # 2.) get X to explain
            # x_to_explain = observations.features.iloc[predictions.index]

            # 3.) run explainer
            # shap_values = explainer.shap_values(x_to_explain)

            # 4.) collect shap_values by index

        # 5.) average/combine/aggregate Shap values when overlapping index

        # 6.) convert to pandas DataFrame

        return pd.DataFrame()

    def feature_dependencies(self, observations: Sample) -> pd.DataFrame:
        # use shap_value_matrix()
        # find correlations
        # return as DataFrame
        pass

    def clustered_feature_importance(self, observations: Sample) -> pd.DataFrame:
        # use shap_value_matrix()
        # cluster it
        # return as DataFrame
        pass
