# coding=utf-8
"""Base classes for wrapper around pipeline returning pandas objects and keeping
track of the column names."""


import logging
from typing import *

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch

from yieldengine.df import DataFrameEstimatorWrapper
from yieldengine.df.predict import DataFrameClassifierWrapper, DataFrameRegressorWrapper
from yieldengine.df.transform import DataFrameTransformerWrapper

log = logging.getLogger(__name__)


class PipelineDF(
    DataFrameTransformerWrapper[Pipeline],
    DataFrameRegressorWrapper[Pipeline],
    DataFrameClassifierWrapper[Pipeline],
):
    """
    Wrapper around `sklearn.pipeline.Pipeline` that returns dataframes.

    :param `**kwargs`: the arguments used to construct the wrapped `Pipeline` object
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._validate_steps()

    def _validate_steps(self) -> None:
        for name, transformer in self._transform_steps():
            if transformer is not None and not isinstance(
                transformer, DataFrameTransformerWrapper
            ):
                raise ValueError(
                    f"expected all transformers to implement DataFrameTransformerWrapper, but "
                    f"step '{name}' is a {type(transformer).__name__}"
                )

    def _transform_steps(self) -> List[Tuple[str, DataFrameTransformerWrapper]]:
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
    def _make_base_estimator(cls, **kwargs) -> Pipeline:
        return Pipeline(**kwargs)

    def _get_columns_original(self) -> pd.Series:
        col_mappings = [
            df_transformer.columns_original
            for _, df_transformer in self._transform_steps()
            if df_transformer is not None
        ]

        if len(col_mappings) == 0:
            _columns_original = _columns_out = self.columns_in
        else:
            _columns_out = col_mappings[-1].index
            _columns_original = col_mappings[-1].values

            # iterate backwards starting from the penultimate item
            for preceding_out_to_original_mapping in col_mappings[-2::-1]:
                # join the original columns of my current transformer on the out columns
                # in the preceding transformer, then repeat
                _columns_original = preceding_out_to_original_mapping.loc[
                    _columns_original
                ].values

        return pd.Series(index=_columns_out, data=_columns_original)

    @property
    def steps(self) -> List[Tuple[str, DataFrameEstimatorWrapper]]:
        """
        The `steps` attribute of the underlying `Pipeline`.

        List of (name, transformer) tuples (transformers implement fit/transform).
        """
        return self.base_transformer.steps

    @property
    def named_steps(self) -> Bunch:
        """
        Read-only attribute to access any step parameter by user given name.

        Keys are step names and values are steps parameters.
        """
        return self.base_transformer.named_steps

    def __len__(self) -> int:
        """The number of steps of the Pipeline."""
        return len(self.base_transformer.steps)

    def __getitem__(self, ind: Union[slice, int, str]) -> DataFrameEstimatorWrapper:
        """
        Return a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `steps` will not change a copy.
        """

        if isinstance(ind, slice):
            base_pipeline = self.base_transformer
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            # noinspection PyTypeChecker
            return self.__class__(
                steps=base_pipeline.steps[ind],
                memory=base_pipeline.memory,
                verbose=base_pipeline.verbose,
            )
        else:
            return self.base_transformer[ind]
