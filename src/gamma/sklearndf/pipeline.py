# coding=utf-8
"""
scikit-learn pipeline implementing the TransformerDF, RegressorDF,
and ClassifierDF interfaces.
"""


import logging
from typing import *
from typing import Any, cast, Generic, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from gamma import ListLike
from gamma.sklearndf import (
    BaseEstimatorDF,
    BasePredictorDF,
    ClassifierDF,
    RegressorDF,
    T_PredictorDF,
    TransformerDF,
)
from gamma.sklearndf._wrapper import (
    ClassifierWrapperDF,
    RegressorWrapperDF,
    TransformerWrapperDF,
)

log = logging.getLogger(__name__)

__all__ = ["PipelineDF", "ModelPipelineDF"]


class PipelineDF(
    ClassifierWrapperDF[Pipeline],
    RegressorWrapperDF[Pipeline],
    TransformerWrapperDF[Pipeline],
):
    """
    Wrapper around :class:`sklearn.pipeline.Pipeline` with dataframes in input and
    output.

    :param `**kwargs`: the arguments used to construct the wrapped
      :class:`~sklearn.pipeline.Pipeline`
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._validate_steps()

    def _validate_steps(self) -> None:
        for name, transformer in self._transform_steps():
            if transformer is not None and not isinstance(transformer, TransformerDF):
                raise ValueError(
                    f"expected all transformers to implement class "
                    f"{TransformerDF.__name__}, but "
                    f"step '{name}' is a {type(transformer).__name__}"
                )

    def _transform_steps(self) -> List[Tuple[str, TransformerDF]]:
        pipeline = self.delegate_estimator

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
    def _make_delegate_estimator(cls, **kwargs) -> Pipeline:
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
    def steps(self) -> List[Tuple[str, BaseEstimatorDF]]:
        """
        The ``steps`` attribute of the underlying :class:`~sklearn.pipeline.Pipeline`.

        List of (name, transformer) tuples (transformers implement fit/transform).
        """
        return self.delegate_estimator.steps

    @property
    def named_steps(self) -> object:
        """
        Read-only attribute to access any step parameter by user given name.

        :return: object with attributes corresponding to the names of the steps
        """
        return self.delegate_estimator.named_steps

    def __len__(self) -> int:
        """The number of steps of the pipeline."""
        return len(self.delegate_estimator.steps)

    def __getitem__(self, ind: Union[slice, int, str]) -> BaseEstimatorDF:
        """
        Return a sub-pipeline or a single estimator in the pipeline

        Indexing with an integer will return an estimator; using a slice
        returns another Pipeline instance which copies a slice of this
        Pipeline. This copy is shallow: modifying (or fitting) estimators in
        the sub-pipeline will affect the larger pipeline and vice-versa.
        However, replacing a value in `steps` will not change a copy.
        """

        if isinstance(ind, slice):
            base_pipeline = self.delegate_estimator
            if ind.step not in (1, None):
                raise ValueError("Pipeline slicing only supports a step of 1")
            # noinspection PyTypeChecker
            return self.__class__(
                steps=base_pipeline.steps[ind],
                memory=base_pipeline.memory,
                verbose=base_pipeline.verbose,
            )
        else:
            return self.delegate_estimator[ind]


class ModelPipelineDF(BaseEstimator, ClassifierDF, RegressorDF, Generic[T_PredictorDF]):
    """
    A data frame enabled pipeline with an optional preprocessing step and a
    mandatory estimator step.

    :param preprocessing: the preprocessing step in the pipeline (defaults to ``None``)
    :param predictor: the delegate estimator used in the pipeline
    :type predictor: :class:`.BasePredictorDF`
    """

    def __init__(
        self, predictor: T_PredictorDF, preprocessing: Optional[TransformerDF] = None
    ) -> None:
        super().__init__()

        if preprocessing is not None and not isinstance(preprocessing, TransformerDF):
            raise TypeError(
                "arg preprocessing expected to be a TransformerDF but is a "
                f"{type(preprocessing).__name__}"
            )
        if not isinstance(predictor, BasePredictorDF):
            raise TypeError(
                "arg predictor expected to be a BasePredictorDF but is a "
                f"{type(predictor).__name__}"
            )

        self.preprocessing = preprocessing
        self.predictor = predictor

    @property
    def delegate_estimator(self) -> T_PredictorDF:
        return self

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "ModelPipelineDF[T_PredictorDF]":
        self.predictor.fit(self._pre_fit_transform(X, y, **fit_params), y, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.preprocessing.is_fitted and self.predictor.is_fitted

    @property
    def columns_in(self) -> pd.Index:
        if self.preprocessing is not None:
            return self.preprocessing.columns_in
        else:
            return self.predictor.columns_in

    # noinspection PyPep8Naming
    def predict(
        self, X: pd.DataFrame, **predict_params
    ) -> Union[pd.Series, pd.DataFrame]:
        return self.predictor.predict(self._pre_transform(X), **predict_params)

    # noinspection PyPep8Naming
    def fit_predict(self, X: pd.DataFrame, y: pd.Series, **fit_params) -> pd.Series:
        return self.predictor.fit_predict(
            self._pre_fit_transform(X, y, **fit_params), y, **fit_params
        )

    # noinspection PyPep8Naming
    def predict_proba(self, X: pd.DataFrame) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        return cast(ClassifierDF, self.predictor).predict_proba(self._pre_transform(X))

    # noinspection PyPep8Naming
    def predict_log_proba(
        self, X: pd.DataFrame
    ) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        return cast(ClassifierDF, self.predictor).predict_log_proba(
            self._pre_transform(X)
        )

    # noinspection PyPep8Naming
    def decision_function(self, X: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        return cast(ClassifierDF, self.predictor).decision_function(
            self._pre_transform(X)
        )

    # noinspection PyPep8Naming
    def score(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        sample_weight: Optional[Any] = None,
    ) -> float:
        if sample_weight is None:
            return self.predictor.score(self._pre_transform(X), y)
        else:
            return self.predictor.score(
                self._pre_transform(X), y, sample_weight=sample_weight
            )

    @property
    def classes(self) -> Optional[ListLike[Any]]:
        return cast(ClassifierDF, self.predictor).classes

    @property
    def n_outputs(self) -> int:
        return self.predictor.n_outputs

    # noinspection PyPep8Naming
    def _pre_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessing is not None:
            return self.preprocessing.transform(X)
        else:
            return X

    # noinspection PyPep8Naming
    def _pre_fit_transform(
        self, X: pd.DataFrame, y: pd.Series, **fit_params
    ) -> pd.DataFrame:
        if self.preprocessing is not None:
            return self.preprocessing.fit_transform(X, y, **fit_params)
        else:
            return X
