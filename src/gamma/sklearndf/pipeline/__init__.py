#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
scikit-learn pipeline implementing the TransformerDF, RegressorDF,
and ClassifierDF interfaces.
"""

import logging
from typing import *

import pandas as pd
from pandas.core.arrays import ExtensionArray
from sklearn.pipeline import FeatureUnion, Pipeline

from gamma.sklearndf import BaseEstimatorDF, TransformerDF
from gamma.sklearndf._wrapper import (
    ClassifierWrapperDF,
    RegressorWrapperDF,
    TransformerWrapperDF,
)
from ._model import *

log = logging.getLogger(__name__)

__all__ = [sym for sym in dir() if sym.endswith("DF")]


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
