from sklearn.model_selection import GridSearchCV
from typing import List
from yieldengine.loading.sample import Sample

# note: unfortunately, sklearn does not expose "BaseSearchCV" from within model_selection, which is the superclass


class ModelSelector:
    def __init__(self, searchers: List[GridSearchCV]) -> None:
        pass

    def train_models(self, sample: Sample) -> None:
        # calls .fit() on each element of self.__searchers
        # needs to convert dataset.drop(columns=target_col_name) and dataset[target_col_name]to numpy array
        pass

    # todo: decide if this should return the list of _models_ OR _gridsearchcv_
    def get_best_models(self) -> List[GridSearchCV]:
        # return a List with GridSearchCV objects in descending order of validation performance

        # i.e. index=0 denotes "best" model, where best model is defined as:
        #       max(gridsearchcv.best_score_)
        #
        # (this is always save to do, because gridsearchcv flips the sign of its score value when
        # the applied scoring method has defined "greater_is_better=False")
        pass
