import pandas as pd
from sklearn.base import BaseEstimator
from yieldengine.modeling.validation import CircularCrossValidator


class ModelInspector:
    def __init__(
        self,
        model: BaseEstimator,
        dataset: pd.DataFrame,
        cv: CircularCrossValidator,
    ) -> None:
        pass

    def get_feature_importance_by_gain(self) -> pd.DataFrame:
        # see: https://lightgbm.readthedocs.io/en/latest/Python-API.html?highlight=importance#lightgbm.LGBMModel.booster_
        # have to assure the current model supports this API - i.e. if not a tree

        # booster_.feature_importance(importance_type='gain', iteration=None)

        # ! have to assure the current model supports this API - i.e. if not a tree

        pass

    def get_feature_importance_by_split(self) -> pd.DataFrame:
        # see: https://lightgbm.readthedocs.io/en/latest/Python-API.html?highlight=importance#lightgbm.LGBMModel.booster_
        # booster_.feature_importance(importance_type='split', iteration=None)

        # ! have to assure the current model supports this API - i.e. if not a tree
        pass

    def get_shap_value_matrix(self) -> pd.DataFrame:
        # for model:
        #   explainer = shap.TreeExplainer(model)
        #   shap_values = explainer.shap_values(X)

        # arrange in matrix, correct/enhance feature names
        # convert to pandas DataFrame

        pass

    def get_feature_dependences(self) -> pd.DataFrame:
        # use get_shap_value_matrix()
        # find correlations
        # return as DataFrame
        pass

    def get_clustered_feature_importance(self) -> pd.DataFrame:
        # use get_shap_value_matrix()
        # cluster it
        # return as DataFrame
        pass

    # internal functions
    def __correct_feature_names(self):
        pass

    def __add_predicted_yield(self):
        pass

    def __add_fold_id(self):
        pass
