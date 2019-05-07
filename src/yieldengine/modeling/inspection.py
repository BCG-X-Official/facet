import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from yieldengine.loading.sample import Sample
from yieldengine.modeling.validation import CircularCrossValidator


class ModelInspector:
    def __init__(
        self,
        model: BaseEstimator,
        cv: CircularCrossValidator,
        preprocessing: Pipeline = None,
    ) -> None:
        pass

    def shap_value_matrix(self, observations: Sample) -> pd.DataFrame:
        # for model:
        #   explainer = shap.TreeExplainer(model)
        #   shap_values = explainer.shap_values(X)

        # arrange in matrix, correct/enhance feature names
        # convert to pandas DataFrame

        pass

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
