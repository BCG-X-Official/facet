"""
Implementation of package ``facet.inspection.shap``.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, List, Optional, Sequence, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc
from pytools.fit import FittableMixin, fitted_only
from pytools.parallelization import ParallelizableMixin

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

T_Model = TypeVar("T_Model")
T_ShapCalculator = TypeVar("T_ShapCalculator", bound="ShapCalculator[Any]")


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


class ShapCalculator(
    FittableMixin[pd.DataFrame],
    ParallelizableMixin,
    Generic[T_Model],
    metaclass=ABCMeta,
):
    """
    Base class for all SHAP calculators.

    A SHAP calculator uses the ``shap`` package to calculate SHAP tensors for all
    observations in a given sample of feature values, then consolidates and aggregates
    results in a data frame.
    """

    #: Name for the feature index (= column index) of the resulting SHAP data frame.
    IDX_FEATURE = "feature"

    #: The explainer factory used to create the SHAP explainer for this calculator.
    explainer_factory: ExplainerFactory[T_Model]

    #: Name of the index that is used to identify multiple outputs for which SHAP
    #: values are calculated. To be overloaded by subclasses.
    MULTI_OUTPUT_INDEX_NAME = "output"

    #: The SHAP values for all observations this calculator has been fitted to.
    shap_: Optional[pd.DataFrame]

    #: The names of the features for which SHAP values were calculated.
    feature_index_: Optional[pd.Index]

    #: The names of the outputs for which SHAP values were calculated.
    output_names_: Optional[Sequence[str]]

    def __init__(
        self,
        explainer_factory: ExplainerFactory[T_Model],
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
        self.explainer_factory = explainer_factory

        # the following attributes are set in fit()
        self.shap_: Optional[pd.DataFrame] = None
        self.feature_index_: Optional[pd.Index] = None
        self.output_names_: Optional[Sequence[str]] = None

    @property
    @abstractmethod
    def interaction_values(self) -> bool:
        """
        ``True`` if this calculator calculates SHAP interaction values, ``False`` if it
        calculates SHAP values.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """[see superclass]"""
        return self.shap_ is not None

    # noinspection PyPep8Naming
    def fit(
        self: T_ShapCalculator, __X: pd.DataFrame, **fit_params: Any
    ) -> T_ShapCalculator:
        """
        Calculate the SHAP values.

        :param __X: the observations for which to calculate SHAP values
        :param fit_params: additional fit parameters (unused)
        :return: self
        :raises ValueError: if the observations are not a valid feature matrix
            for this calculator
        """

        # reset fit in case we get an exception along the way
        self._reset_fit()

        # validate the feature matrix
        self.validate_features(__X)

        self.feature_index_ = __X.columns.rename(ShapCalculator.IDX_FEATURE)
        self.output_names_ = self.get_output_names()

        # explain all observations using the model, resulting in a matrix of
        # SHAP values for each observation and feature
        shap_df: pd.DataFrame = self._calculate_shap(
            features=__X, explainer=self._get_explainer(__X)
        )

        # re-order the observation index to match the sequence in the original
        # training sample

        n_levels = shap_df.index.nlevels
        assert 1 <= n_levels <= 2
        assert shap_df.index.names[0] == __X.index.name

        self.shap_ = shap_df.reindex(
            index=__X.index.intersection(
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
    def get_output_names(self) -> List[str]:
        """
        :return: a name for each of the outputs explained by this calculator
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

    @abstractmethod
    def validate_features(self, features: pd.DataFrame) -> None:
        """
        Check that the given feature matrix is valid for this calculator.

        :param features: the feature matrix to validate
        :raise ValueError: if the feature matrix is not compatible with this
            calculator
        """
        pass

    def _reset_fit(self) -> None:
        # set this calculator to its initial unfitted state
        self.shap_ = None
        self.feature_index_ = None
        self.output_names_ = None

    @abstractmethod
    def _get_explainer(self, features: pd.DataFrame) -> BaseExplainer:
        pass

    def _calculate_shap(
        self, *, features: pd.DataFrame, explainer: BaseExplainer
    ) -> pd.DataFrame:
        if features.isna().values.any():
            log.warning(
                "preprocessed features passed to SHAP explainer include NaN values; "
                "try to change preprocessing to impute all NaN values"
            )

        multi_output_index_name = self.MULTI_OUTPUT_INDEX_NAME
        multi_output_names = self.get_output_names()
        assert self.feature_index_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        features_out = self.feature_index_

        # calculate the shap values, and ensure the result is a list of arrays
        shap_values: List[npt.NDArray[np.float_]] = self._convert_shap_tensors_to_list(
            shap_tensors=(
                explainer.shap_interaction_values(features)
                if self.interaction_values
                else explainer.shap_values(features)
            ),
            n_outputs=len(multi_output_names),
        )

        # convert to a data frame per output (different logic depending on whether
        # we have a regressor or a classifier, implemented by method
        # shap_matrix_for_split_to_df_fn)

        shap_values_df_per_output: List[pd.DataFrame] = self._convert_raw_shap_to_df(
            shap_values, features.index, features_out
        )

        # if we have a single output, return the data frame for that output;
        # else, add a top level to the column index indicating each output

        if len(shap_values_df_per_output) == 1:
            return shap_values_df_per_output[0]
        else:
            return pd.concat(
                shap_values_df_per_output,
                axis=1,
                keys=multi_output_names,
                names=[multi_output_index_name, features_out.name],
            )

    def _convert_shap_tensors_to_list(
        self,
        *,
        shap_tensors: Union[npt.NDArray[np.float_], List[npt.NDArray[np.float_]]],
        n_outputs: int,
    ) -> List[npt.NDArray[np.float_]]:
        def _validate_shap_tensor(_t: npt.NDArray[np.float_]) -> None:
            if np.isnan(np.sum(_t)):
                raise AssertionError(
                    "Output of SHAP explainer includes NaN values. "
                    "This should not happen; consider initialising the "
                    "LearnerInspector with an ExplainerFactory that has a different "
                    "configuration, or that makes SHAP explainers of a different type."
                )

        if isinstance(shap_tensors, List):
            for shap_tensor in shap_tensors:
                _validate_shap_tensor(shap_tensor)
        else:
            _validate_shap_tensor(shap_tensors)
            shap_tensors = [shap_tensors]

        if n_outputs != len(shap_tensors):
            raise AssertionError(
                f"count of SHAP tensors (n={len(shap_tensors)}) "
                f"should match number of outputs (n={n_outputs})"
            )

        return shap_tensors

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
class ShapValuesCalculator(
    ShapCalculator[T_Model], Generic[T_Model], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP contribution values.
    """

    @property
    def interaction_values(self) -> bool:
        """[see superclass]"""
        return False

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


@inheritdoc(match="""[see superclass]""")
class ShapInteractionValuesCalculator(
    ShapCalculator[T_Model], Generic[T_Model], metaclass=ABCMeta
):
    """
    Base class for calculating SHAP interaction values.
    """

    @property
    def interaction_values(self) -> bool:
        """[see superclass]"""
        return True

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
            self.shap_ is not None and self.feature_index_ is not None
        ), ASSERTION__CALCULATOR_IS_FITTED

        n_observations = len(self.shap_)
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


__tracker.validate()
