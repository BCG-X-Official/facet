# coding=utf-8
"""
scikit-learn pipeline implementing the TransformerDF, RegressorDF,
and ClassifierDF interfaces.
"""

import logging
from typing import *
from typing import Any, cast, Generic, List, Optional, Union

import pandas as pd
from pandas.core.arrays import ExtensionArray
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline

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

__all__ = ["FeatureUnionDF", "PipelineDF", "ModelPipelineDF"]


class PipelineDF(
    ClassifierWrapperDF[Pipeline],
    RegressorWrapperDF[Pipeline],
    TransformerWrapperDF[Pipeline],
):
    """
    Wrapper around :class:`sklearn.pipeline.Pipeline` with data frames in input and
    output.

    :param `**kwargs`: the arguments used to construct the wrapped
      :class:`~sklearn.pipeline.Pipeline`
    """

    PASSTHROUGH = "passthrough"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # ensure that all steps support data frames, and that all except the last
        # step are data frame transformers

        steps = self.steps

        if len(steps) == 0:
            return

        for name, transformer in steps[:-1]:
            if not (
                self._is_passthrough(transformer)
                or isinstance(transformer, TransformerDF)
            ):
                raise ValueError(
                    f'expected step "{name}" to contain a '
                    f"{TransformerDF.__name__}, but found an instance of "
                    f"{type(transformer).__name__}"
                )

        final_step = steps[-1]
        final_estimator = final_step[1]
        if not (
            self._is_passthrough(final_estimator)
            or isinstance(final_estimator, BaseEstimatorDF)
        ):
            raise ValueError(
                f'expected final step "{final_step[0]}" to contain a '
                f"{BaseEstimatorDF.__name__}, but found an instance of "
                f"{type(final_estimator).__name__}"
            )

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> Pipeline:
        return Pipeline(*args, **kwargs)

    @property
    def steps(self) -> List[Tuple[str, BaseEstimatorDF]]:
        """
        The ``steps`` attribute of the underlying :class:`~sklearn.pipeline.Pipeline`.

        List of (name, transformer) tuples (transformers implement fit/transform).
        """
        return self.delegate_estimator.steps

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

            return self.__class__(
                steps=base_pipeline.steps[ind],
                memory=base_pipeline.memory,
                verbose=base_pipeline.verbose,
            )
        else:
            return self.delegate_estimator[ind]

    @staticmethod
    def _is_passthrough(estimator: Union[BaseEstimatorDF, str, None]) -> bool:
        # return True if the estimator is a "passthrough" (i.e. identity) transformer
        # in the pipeline
        return estimator is None or estimator == PipelineDF.PASSTHROUGH

    def _transformer_steps(self) -> Iterator[Tuple[str, TransformerDF]]:
        # make an iterator of all transform steps, i.e. excluding the final step
        # in case it is not a transformer
        # excludes steps whose transformer is `None` or `"passthrough"`

        def _iter_not_none(
            transformer_steps: Sequence[Tuple[str, BaseEstimatorDF]]
        ) -> Iterator[Tuple[str, TransformerDF]]:
            return (
                (name, cast(TransformerDF, transformer))
                for name, transformer in transformer_steps
                if not self._is_passthrough(transformer)
            )

        steps = self.steps

        if len(steps) == 0:
            return iter([])

        final_estimator = steps[-1][1]

        if isinstance(final_estimator, TransformerDF):
            return _iter_not_none(steps)
        else:
            return _iter_not_none(steps[:-1])

    def _get_columns_original(self) -> pd.Series:
        col_mappings = [
            df_transformer.columns_original
            for _, df_transformer in self._transformer_steps()
        ]

        if len(col_mappings) == 0:
            _columns_out: pd.Index = self.columns_in
            _columns_original: Union[
                pd.np.ndarray, ExtensionArray
            ] = _columns_out.values
        else:
            _columns_out: pd.Index = col_mappings[-1].index
            _columns_original: Union[pd.np.ndarray, ExtensionArray] = col_mappings[
                -1
            ].values

            # iterate backwards starting from the penultimate item
            for preceding_out_to_original_mapping in col_mappings[-2::-1]:
                # join the original columns of my current transformer on the out columns
                # in the preceding transformer, then repeat
                _columns_original = preceding_out_to_original_mapping.loc[
                    _columns_original
                ].values

        return pd.Series(index=_columns_out, data=_columns_original)

    def _get_columns_out(self) -> pd.Index:
        for _, transformer in reversed(self.steps):
            if isinstance(transformer, TransformerDF):
                return transformer.columns_out

        return self.columns_in


class FeatureUnionDF(TransformerWrapperDF[FeatureUnion]):
    """
    Wraps :class:`sklearn.pipeline.FeatureUnion`;
    accepts and returns data frames.
    """

    @classmethod
    def _make_delegate_estimator(cls, *args, **kwargs) -> FeatureUnion:
        return FeatureUnion(*args, **kwargs)

    @staticmethod
    def _prepend_columns_out(columns_out: pd.Index, name_prefix: str) -> pd.Index:
        return pd.Index(data=f"{name_prefix}__" + columns_out.astype(str))

    def _get_columns_original(self) -> pd.Series:
        # concatenate output->input mappings from all included transformers other than
        # ones stated as `None` or `"drop"` or any other string

        # prepend the name of the transformer so the resulting feature name is
        # `<name>__<output column of sub-transformer>

        def _prepend_columns_original(
            columns_original: pd.Series, name_prefix: str
        ) -> pd.Series:
            return pd.Series(
                data=columns_original.values,
                index=self._prepend_columns_out(
                    columns_out=columns_original.index, name_prefix=name_prefix
                ),
            )

        # noinspection PyProtectedMember
        return pd.concat(
            objs=(
                _prepend_columns_original(
                    columns_original=transformer.columns_original, name_prefix=name
                )
                for name, transformer, _ in self.delegate_estimator._iter()
            )
        )

    def _get_columns_out(self) -> pd.Index:
        # concatenate output columns from all included transformers other than
        # ones stated as `None` or `"drop"` or any other string

        # prepend the name of the transformer so the resulting feature name is
        # `<name>__<output column of sub-transformer>

        # noinspection PyProtectedMember
        indices = [
            self._prepend_columns_out(
                columns_out=transformer.columns_out, name_prefix=name
            )
            for name, transformer, _ in self.delegate_estimator._iter()
        ]

        if len(indices) == 0:
            return pd.Index()
        else:
            return indices[0].append(other=indices[1:])


#
# GAMMA custom pipelines
#


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

    # noinspection PyPep8Naming
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> "ModelPipelineDF[T_PredictorDF]":
        self.predictor.fit(self._pre_fit_transform(X, y, **fit_params), y, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.preprocessing.is_fitted and self.predictor.is_fitted

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

    def _get_columns_in(self) -> pd.Index:
        if self.preprocessing is not None:
            return self.preprocessing.columns_in
        else:
            return self.predictor.columns_in

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
