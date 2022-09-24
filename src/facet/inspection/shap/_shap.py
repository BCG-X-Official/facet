"""
Implementation of package ``facet.inspection.shap``.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin, fitted_only
from pytools.parallelization import ParallelizableMixin

from facet.data import Sample
from facet.inspection._explainer import BaseExplainer, ExplainerFactory

log = logging.getLogger(__name__)

__all__ = [
    "ShapCalculator",
    "ShapInteractionValuesCalculator",
    "ShapValuesCalculator",
]

#
# Type variables
#

T_ShapCalculator = TypeVar("T_ShapCalculator", bound="ShapCalculator")


#
# Constants
#

ASSERTION__CALCULATOR_IS_FITTED = "calculator is fitted"


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class ShapCalculator(FittableMixin[Sample], ParallelizableMixin, metaclass=ABCMeta):
    """
    Base class for all SHAP calculators.

    A SHAP calculator uses the ``shap`` package to calculate SHAP tensors for all
    observations in a given sample, then consolidates and aggregates results
    in a data frame.
    """

    #: constant for "mean" aggregation method, to be passed as arg ``aggregation``
    #: to :class:`.ShapCalculator` methods that implement it
    AGG_MEAN = "mean"

    #: constant for "std" aggregation method, to be passed as arg ``aggregation``
    #: to :class:`.ShapCalculator` methods that implement it
    AGG_STD = "std"

    #: name of index level indicating the split ID
    IDX_SPLIT = "split"

    def __init__(
        self,
        explainer_factory: ExplainerFactory,
        *,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self._explainer_factory = explainer_factory

        # the following attributes are set in fit()
        self.shap_: Optional[pd.DataFrame] = None
        self.feature_index_: Optional[pd.Index] = None
        self.output_names_: Optional[Sequence[str]] = None
        self.sample_: Optional[Sample] = None

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.shap_ is not None

    def fit(
        self: T_ShapCalculator, __sample: Sample, **fit_params: Any
    ) -> T_ShapCalculator:
        """
        Calculate the SHAP values.

        :param __sample: the observations for which to calculate SHAP values
        :param fit_params: additional fit parameters (unused)
        :return: self
        """

        # reset fit in case we get an exception along the way
        self.shap_ = None

        self.feature_index_ = self.get_feature_names()
        self.output_names_ = self._get_output_names(__sample)
        self.sample_ = __sample

        # calculate shap values and re-order the observation index to match the
        # sequence in the original training sample
        shap_df: pd.DataFrame = self._get_shap(__sample)

        n_levels = shap_df.index.nlevels
        assert 1 <= n_levels <= 2
        assert shap_df.index.names[0] == __sample.index.name

        self.shap_ = shap_df.reindex(
            index=__sample.index.intersection(
                (
                    shap_df.index
                    if n_levels == 1
                    else cast(pd.MultiIndex, shap_df.index).levels[0]
                ),
                sort=False,
            ),
            level=0,
            copy=False,
        )

        return self

    @abstractmethod
    def get_feature_names(self) -> pd.Index:
        """
        Get the feature names for which SHAP values are calculated.

        :return: the feature names
        """
        pass

    @abstractmethod
    def get_shap_values(self) -> pd.DataFrame:
        """
        The resulting shap values, per observation and feature, as a data frame.

        :return: SHAP contribution values with shape
            (n_observations, n_outputs * n_features)
        """

    @abstractmethod
    def get_shap_interaction_values(self) -> pd.DataFrame:
        """
        Get the resulting shap interaction values as a data frame.

        :return: SHAP contribution values with shape
            (n_observations * n_features, n_outputs * n_features)
        :raise TypeError: this SHAP calculator does not support interaction values
        """

    @staticmethod
    @abstractmethod
    def get_multi_output_type() -> str:
        """
        :return: a category name for the dimensions represented by multiple outputs
        """

    @abstractmethod
    def get_multi_output_names(self, sample: Sample) -> List[str]:
        """
        :return: a name for each of the outputs
        """
        pass

    @abstractmethod
    def preprocess_features(self, sample: Sample) -> pd.DataFrame:
        """
        Preprocess the features in the sample prior to SHAP calculation.

        :param sample:
        :return:
        """
        pass

    @abstractmethod
    def _get_shap(self, sample: Sample) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_output_names(self, sample: Sample) -> Sequence[str]:
        return self.get_multi_output_names(sample)

    @abstractmethod
    def _calculate_shap(
        self, *, sample: Sample, explainer: BaseExplainer
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def _convert_shap_tensors_to_list(
        self,
        *,
        shap_tensors: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]],
        n_outputs: int,
    ) -> List[npt.NDArray[np.float_]]:
        pass

    @staticmethod
    @abstractmethod
    def _convert_raw_shap_to_df(
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observations: pd.Index,
        features_in_split: pd.Index,
    ) -> List[pd.DataFrame]:
        """
        Convert the SHAP tensors for a single split to a data frame.

        :param raw_shap_tensors: the raw values returned by the SHAP explainer
        :param observations: the ids used for indexing the explained observations
        :param features_in_split: the features in the current split,
            explained by the SHAP explainer
        :return: SHAP values of a single split as data frame
        """
        pass


@inheritdoc(match="""[see superclass]""")
class ShapValuesCalculator(ShapCalculator, metaclass=ABCMeta):
    """
    Base class for calculating SHAP contribution values.
    """

    @fitted_only
    def get_shap_values(self) -> pd.DataFrame:
        """[see superclass]"""
        return self.shap_

    def get_shap_interaction_values(self) -> pd.DataFrame:
        """
        Not implemented.

        :return: (never returns anything)
        :raise TypeError: SHAP interaction values are not supported - always raised
        """
        raise TypeError(
            f"{type(self).__name__}"
            f".{ShapValuesCalculator.get_shap_interaction_values.__name__}() "
            "is not defined"
        )

    def _calculate_shap(
        self, *, sample: Sample, explainer: BaseExplainer
    ) -> pd.DataFrame:
        x = self.preprocess_features(sample=sample)

        if x.isna().values.any():
            log.warning(
                "preprocessed sample passed to SHAP explainer contains NaN values; "
                "try to change preprocessing to impute all NaN values"
            )

        multi_output_type = self.get_multi_output_type()
        multi_output_names = self.get_multi_output_names(sample=sample)
        assert self.feature_index_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        features_out = self.feature_index_

        # calculate the shap values, and ensure the result is a list of arrays
        shap_values: List[npt.NDArray[np.float_]] = self._convert_shap_tensors_to_list(
            shap_tensors=explainer.shap_values(x), n_outputs=len(multi_output_names)
        )

        # convert to a data frame per output (different logic depending on whether
        # we have a regressor or a classifier, implemented by method
        # shap_matrix_for_split_to_df_fn)
        shap_values_df_per_output: List[pd.DataFrame] = [
            shap.reindex(columns=features_out, copy=False, fill_value=0.0)
            for shap in self._convert_raw_shap_to_df(shap_values, x.index, x.columns)
        ]

        # if we have a single output, return the data frame for that output;
        # else, add a top level to the column index indicating each output

        if len(shap_values_df_per_output) == 1:
            return shap_values_df_per_output[0]
        else:
            return pd.concat(
                shap_values_df_per_output,
                axis=1,
                keys=multi_output_names,
                names=[multi_output_type, features_out.name],
            )


@inheritdoc(match="""[see superclass]""")
class ShapInteractionValuesCalculator(ShapCalculator, metaclass=ABCMeta):
    """
    Base class for calculating SHAP interaction values.
    """

    @fitted_only
    def get_shap_values(self) -> pd.DataFrame:
        """[see superclass]"""

        assert self.shap_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        return self.shap_.groupby(level=0).sum()

    @fitted_only
    def get_shap_interaction_values(self) -> pd.DataFrame:
        """[see superclass]"""

        assert self.shap_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        return self.shap_

    @fitted_only
    def get_diagonals(self) -> pd.DataFrame:
        """
        The get_diagonals of all SHAP interaction matrices, of shape
        (n_observations, n_outputs * n_features).

        :return: SHAP interaction values with shape
            (n_observations * n_features, n_outputs * n_features), i.e., for each
            observation and output we get the feature interaction values of size
            n_features * n_features.
        """

        assert (
            self.shap_ is not None
            and self.sample_ is not None
            and self.feature_index_ is not None
        ), ASSERTION__CALCULATOR_IS_FITTED

        n_observations = len(self.sample_)
        n_features = len(self.feature_index_)
        interaction_matrix = self.shap_

        return pd.DataFrame(
            np.diagonal(
                interaction_matrix.values.reshape(
                    (n_observations, n_features, -1, n_features)
                    # observations x features x outputs x features
                ),
                axis1=1,
                axis2=3,
            ).reshape((n_observations, -1)),
            # observations x (outputs * features)
            index=cast(pd.MultiIndex, interaction_matrix.index).levels[0],
            columns=interaction_matrix.columns,
        )

    def _calculate_shap(
        self, *, sample: Sample, explainer: BaseExplainer
    ) -> pd.DataFrame:
        x = self.preprocess_features(sample=sample)

        multi_output_type = self.get_multi_output_type()
        multi_output_names = self.get_multi_output_names(sample)
        assert self.feature_index_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        features_out = self.feature_index_

        # calculate the shap interaction values; ensure the result is a list of arrays
        shap_interaction_tensors: List[
            npt.NDArray[np.float_]
        ] = self._convert_shap_tensors_to_list(
            shap_tensors=explainer.shap_interaction_values(x),
            n_outputs=len(multi_output_names),
        )

        interaction_matrix_per_output: List[pd.DataFrame] = [
            im.reindex(
                index=pd.MultiIndex.from_product(
                    iterables=(x.index, features_out),
                    names=(x.index.name, features_out.name),
                ),
                columns=features_out,
                copy=False,
                fill_value=0.0,
            )
            for im in self._convert_raw_shap_to_df(
                shap_interaction_tensors, x.index, x.columns
            )
        ]

        # if we have a single output, use the data frame for that output;
        # else, concatenate the values data frame for all outputs horizontally
        # and add a top level to the column index indicating each output
        if len(interaction_matrix_per_output) == 1:
            return interaction_matrix_per_output[0]
        else:
            return pd.concat(
                interaction_matrix_per_output,
                axis=1,
                keys=multi_output_names,
                names=[multi_output_type, features_out.name],
            )


__tracker.validate()
