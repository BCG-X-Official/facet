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
    ) -> BaseEstimator:
        if isinstance(estimators, BaseEstimator):
            # use sklearn.pipeline.make_pipeline to build a trivial pipeline
            # containing only the given estimator
            return estimators
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
            return FeatureUnion(transformer_list=estimator_steps)

    def make_pipeline(
        self, estimators: Union[BaseEstimator, Iterable[BaseEstimator]]
    ) -> Pipeline:
        return Pipeline(
            [
                (
                    ModelPipelineFactory.STEP_ESTIMATORS,
                    self._make_estimator_step(estimators=estimators),
                )
            ]
        )

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

    def __init__(self, preprocessing_step: BaseEstimator, memory: bool = True):
        super().__init__()
        # noinspection PyUnresolvedReferences
        if not hasattr(preprocessing_step, "transform") or not isinstance(
            preprocessing_step.transform, Callable
        ):
            raise ValueError(
                "attribute preprocessing_step needs to implement "
                "function transform()"
            )
        self._preprocessing = preprocessing_step
        self._memory = memory

    @property
    def preprocessing_step(self) -> BaseEstimator:
        return self._preprocessing

    @property
    def memory(self) -> bool:
        return self._memory

    def make_pipeline(
        self, estimators: Union[BaseEstimator, Iterable[BaseEstimator]]
    ) -> Pipeline:
        return Pipeline(
            [
                self.preprocessing_step,
                super()._make_estimator_step(estimators=estimators),
            ],
            memory=f"preprocessing<{id(self)}>" if self.memory else None,
        )


class SimplePreprocessingPipelineFactory(PreprocessingModelPipelineFactory):
    def __init__(
        self,
        mean_impute: Iterable[str] = None,
        one_hot_encode: Iterable[str] = None,
        memory: bool = True,
    ):
        """

        :param mean_impute: list of columns to impute or None
        :param one_hot_encode: list of (categorical) columns to encode or None
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
