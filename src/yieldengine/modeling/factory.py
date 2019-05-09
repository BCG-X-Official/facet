from typing import *

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder


class ModelPipelineFactory:
    STEP_ESTIMATORS = "estimators"

    def __init__(self) -> None:
        super().__init__()

    def _make_estimator_step(
        self, estimators: Union[BaseEstimator, Iterable[BaseEstimator]]
    ) -> Tuple[str, BaseEstimator]:
        if isinstance(estimators, BaseEstimator):
            return ModelPipelineFactory.STEP_ESTIMATORS, estimators
        else:
            # generate a list of (name, obj) tuples for all estimators
            #   name: e0, e1, e2, ...
            #   obj: the estimator wrapped in a ModelTransformer() object,
            #   to be feature union compliant
            estimator_steps = [
                (f"e{i}", PreprocessingModelPipelineFactory.ModelTransformer(estimator))
                for i, estimator in enumerate(estimators)
            ]

            # with the above created list of ModelTransformers, create FeatureUnion()
            # benefit: can be evaluated in parallel and pre-processing is shared
            return (
                ModelPipelineFactory.STEP_ESTIMATORS,
                FeatureUnion(transformer_list=estimator_steps),
            )

    def make_pipeline(
        self, estimators: Union[BaseEstimator, Iterable[BaseEstimator]]
    ) -> Pipeline:
        return Pipeline([self._make_estimator_step(estimators=estimators)])

    class ModelTransformer(TransformerMixin):
        """ helper class to view a model (Estimator/GridSearchCV) as a Transformer """

        def __init__(self, model) -> None:
            self.model = model

        def fit(
            self, *args, **kwargs
        ) -> "PreprocessingModelPipelineFactory.ModelTransformer":
            self.model.fit(*args, **kwargs)
            return self

        def transform(self, X, **transform_params) -> pd.DataFrame:
            return pd.DataFrame(self.model.predict(X))


class PreprocessingModelPipelineFactory(ModelPipelineFactory):
    __slots__ = ["_preprocessing", "_memory"]

    def __init__(
        self,
        preprocessing_step: Tuple[str, BaseEstimator],
        memory: Union[str, object] = None,
    ):
        super().__init__()
        # noinspection PyUnresolvedReferences
        if not hasattr(preprocessing_step[1], "transform") or not isinstance(
            preprocessing_step[1].transform, Callable
        ):
            raise ValueError(
                "attribute preprocessing_step needs to implement "
                "function transform()"
            )
        self._preprocessing = preprocessing_step
        self._memory = memory

    @property
    def preprocessing_step(self) -> Tuple[str, BaseEstimator]:
        return self._preprocessing

    def make_pipeline(
        self, estimators: Union[BaseEstimator, Iterable[BaseEstimator]]
    ) -> Pipeline:
        return Pipeline(
            steps=[
                self.preprocessing_step,
                super()._make_estimator_step(estimators=estimators),
            ],
            memory=self._memory,
        )


class SimplePreprocessingPipelineFactory(PreprocessingModelPipelineFactory):
    def __init__(
        self,
        mean_impute: Iterable[str] = None,
        one_hot_encode: Iterable[str] = None,
        memory: Union[str, object] = None,
    ):
        """

        :param mean_impute: list of columns to impute or None
        :param one_hot_encode: list of (categorical) columns to encode or None
        :param memory: string or object to be passed to the Pipeline's memory parameter
        """

        transformations: List[Tuple[str, TransformerMixin, Iterable[str]]] = list()

        if mean_impute is not None:
            transformations.append(
                ("impute", SimpleImputer(strategy="mean"), mean_impute)
            )

        if one_hot_encode is not None:
            transformations.append(
                (
                    "one-hot-encode",
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    one_hot_encode,
                )
            )

        super().__init__(
            preprocessing_step=ColumnTransformer(transformations), memory=memory
        )
