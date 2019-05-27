# coding=utf-8

import logging
from typing import *

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from yieldengine.feature.transform import DataFrameTransformer

log = logging.getLogger(__name__)


class PipelineDF(DataFrameTransformer[Pipeline]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._validate_steps()

    def _validate_steps(self) -> None:
        for name, transformer in self._transform_steps():
            if transformer is not None and not isinstance(
                transformer, DataFrameTransformer
            ):
                raise ValueError(
                    f"expected all transformers to implement DataFrameTransformer, but "
                    f"step '{name}' is a {type(transformer).__name__}"
                )

    def _transform_steps(self) -> List[Tuple[str, DataFrameTransformer]]:
        pipeline = self.base_transformer

        steps = pipeline.steps

        if len(steps) == 0:
            return []

        estimator = steps[-1][1]

        if estimator is not None and (
            hasattr(estimator, "fit_transform") or hasattr(estimator, "transform")
        ):
            transform_steps = steps
        else:
            transform_steps = steps[:-1]
        return transform_steps

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> Pipeline:
        return Pipeline(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        transform_steps = self._transform_steps()
        if len(transform_steps) == 0:
            return self.columns_in
        else:
            return transform_steps[-1][1].columns_out

    def _get_columns_original(self) -> pd.Series:
        col_mappings = [
            df_transformer.columns_original
            for _, df_transformer in self._transform_steps()
        ]

        _columns_original = col_mappings[-1].values

        for sub_mapping in col_mappings[:-1:-1]:
            # join the original columns of my current transformer on the out columns in
            # the preceding transformer, then repeat
            _columns_original = sub_mapping.loc[_columns_original].values

        return pd.Series(
            index=col_mappings[-1].index,
            data=_columns_original,
            name=DataFrameTransformer.F_COLUMN_ORIGINAL,
        )

    @property
    def steps(self) -> Sequence[Tuple[str, Union[DataFrameTransformer, BaseEstimator]]]:
        return self.base_transformer.steps

    # noinspection PyPep8Naming
    def predict(self, X: pd.DataFrame, **predict_params):
        return self.base_transformer.predict(X, **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y=pd.Series, **fit_params):
        return self.base_transformer.fit_predict(X, y, **fit_params)

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame):
        return self.base_transformer.predict_proba(X)

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame):
        return self.base_transformer.decision_function(X)

    # noinspection PyPep8Naming
    def predict_log_proba(self, X: pd.DataFrame):
        return self.base_transformer.predict_log_proba(X)

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ):
        return self.base_transformer.score(X, y, sample_weight)

    def __len__(self) -> int:
        """
        @returns the length of the Pipeline
        """
        return len(self.base_transformer.steps)

    def __getitem__(self, ind: Union[slice, int, str]) -> DataFrameTransformer:
        """Returns a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `step` will not affect a copy.
        """

        if isinstance(ind, slice):
            base_pipeline = self.base_transformer
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            return self.__class__(
                steps=base_pipeline.steps[ind],
                memory=base_pipeline.memory,
                verbose=base_pipeline.verbose,
            )
        else:
            return self.base_transformer[ind]
