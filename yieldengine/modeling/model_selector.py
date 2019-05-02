from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List
from yieldengine.loading.sample import Sample
import pandas as pd

# note: unfortunately, sklearn does not expose "BaseSearchCV" from within model_selection, which is the superclass


class ModelSelector:
    def __init__(
        self, searchers: List[GridSearchCV], preprocessing: Pipeline = None
    ) -> None:
        self.__searchers = searchers
        self.__preprocessing = preprocessing
        self.__pipeline = None

    def train_models(self, sample: Sample) -> None:
        class ModelTransformer(TransformerMixin):
            def __init__(self, model):
                self.model = model

            def fit(self, *args, **kwargs):
                self.model.fit(*args, **kwargs)
                return self

            def transform(self, X, **transform_params):
                return pd.DataFrame(self.model.predict(X))

        # generate a list of (name, obj) tuples for all given estimators (=GridSearchCV objs.):
        #   name: simply e0, e1, e2, ...
        #   obj: the GridSearchCV wrapped in a ModelTransformer() object, to be feature union compliant
        estimator_steps = [
            (f"e{i}", ModelTransformer(estimator))
            for i, estimator in enumerate(self.__searchers)
        ]

        # with the above created list of ModelTransformers, create FeatureUnion()
        # benefit: can be evaluated in parallel and pre-processing is shared
        est_feature_union = (
            "estimators",
            FeatureUnion(transformer_list=estimator_steps),
        )

        # if pre-processing pipeline was given, insert it into the pipeline:
        if self.__preprocessing is not None:
            self.__pipeline = Pipeline(
                [("preprocessing", self.__preprocessing), est_feature_union]
            )
        else:
            # if pre-processing pipeline was not given, create a minimal Pipeline with just the estimators
            self.__pipeline = Pipeline([est_feature_union])

        self.__pipeline.fit(sample.feature_data, sample.target_data)

    def get_best_models(self) -> List[GridSearchCV]:
        l = self.__searchers.copy()

        # always sort best_score_ in descending fashion, since gridsearchcv flips the sign of its score value when
        # the applied scoring method has defined "greater_is_better=False")
        l.sort(key=lambda x: x.best_score_ * -1)
        return l

    def summary_string(self, limit: int = 25) -> str:
        return "\n".join(
            [
                f" Rank {i + 1}: {m.estimator.__class__}, Score: {-1 * m.best_score_}, Params: {m.best_params_}"
                for i, m in enumerate(self.get_best_models())
                if i < limit
            ]
        )
