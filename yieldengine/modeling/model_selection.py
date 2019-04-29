import pandas as pd
from sklearn.model_selection import GridSearchCV
from typing import List

# note: unfortunately, sklearn does not expose "BaseSearchCV" from within model_selection, which is the superclass


class ModelSelector:
    def __init__(self, gridsearchcv_list: List[GridSearchCV]):
        pass

    def train_models(self, dataset: pd.DataFrame, target_col_name: str):
        # calls .fit() on each element of self.__gridsearchcv_list
        # needs to convert dataset.drop(columns=target_col_name) and dataset[target_col_name]to numpy array
        pass

    def get_global_best_model(self) -> GridSearchCV:
        # find global best score of "best_score_" values within self.__gridsearchcv_list
        pass

    def get_n_best_models(self, n: int) -> List[GridSearchCV]:
        # find #n models ranked by "best_score_"
        pass

    def get_all_tuned_models(self) -> List[GridSearchCV]:
        # extract all models out of the GridSearchCV objects
        # using: best_estimator_
        pass
