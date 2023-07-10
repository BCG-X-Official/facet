"""
Implementation of package ``facet.inspection.shap``.
"""

import logging
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, List, Optional, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker
from pytools.fit import FittableMixin, fitted_only
from pytools.parallelization import ParallelizableMixin

from ...explanation.base import BaseExplainer, ExplainerFactory
from ...explanation.parallel import ParallelExplainer

log = logging.getLogger(__name__)

__all__ = [
    "ShapCalculator",
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

    #: Name of the index that is used to identify multiple outputs for which SHAP
    #: values are calculated. To be overloaded by subclasses.
    MULTI_OUTPUT_INDEX_NAME = "output"

    #: The model for which to calculate SHAP values.
    model: T_Model

    #: The explainer factory used to create the SHAP explainer for this calculator.
    explainer_factory: ExplainerFactory[T_Model]

    #: The SHAP values for all observations this calculator has been fitted to.
    shap_: Optional[pd.DataFrame]

    #: The names of the features for which SHAP values were calculated.
    feature_index_: Optional[pd.Index]

    # defined in superclass, repeated here for Sphinx:
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

    def __init__(
        self,
        model: T_Model,
        *,
        explainer_factory: ExplainerFactory[T_Model],
        interaction_values: bool,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param model: the model for which to calculate SHAP values
        :param explainer_factory: the explainer factory used to create the SHAP
            explainer for this calculator
        :param interaction_values: if ``True``, calculate SHAP interaction values,
            otherwise calculate SHAP values
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.model = model
        self.explainer_factory = explainer_factory
        self.interaction_values = interaction_values

        # the following attributes are set in fit()
        self.shap_: Optional[pd.DataFrame] = None
        self.feature_index_: Optional[pd.Index] = None

    __init__.__doc__ = cast(str, __init__.__doc__) + cast(
        str, ParallelizableMixin.__init__.__doc__
    )

    @property
    @abstractmethod
    def input_names(self) -> Optional[List[str]]:
        """
        The names of the inputs explained by this SHAP calculator, or ``None`` if
        no names are defined.
        """

    @property
    @abstractmethod
    def output_names(self) -> List[str]:
        """
        The names of the outputs explained by this SHAP calculator.
        """

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

        # explain all observations using the model, resulting in a matrix of
        # SHAP values for each observation and feature
        shap_df: pd.DataFrame = self._calculate_shap(
            features=__X, explainer=self._make_explainer(__X)
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

    @property
    @fitted_only
    def shap_values(self) -> pd.DataFrame:
        r"""
        The SHAP values per observation and feature, with shape
        :math:`(n_\mathrm{observations}, n_\mathrm{outputs} \cdot n_\mathrm{features})`
        """

        assert self.shap_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        if self.interaction_values:
            return self.shap_.groupby(level=0, sort=False).sum()
        else:
            return self.shap_

    @property
    @fitted_only
    def shap_interaction_values(self) -> pd.DataFrame:
        r"""
        The SHAP interaction values per observation and feature pair, with shape
        :math:`(n_\mathrm{observations} \cdot n_\mathrm{features}, n_\mathrm{outputs}
        \cdot n_\mathrm{features})`

        :raise AttributeError: this SHAP calculator does not support interaction values
        """
        if self.interaction_values:
            assert self.shap_ is not None, ASSERTION__CALCULATOR_IS_FITTED
            return self.shap_
        else:
            raise AttributeError("interaction values are not supported")

    @property
    @fitted_only
    def main_effects(self) -> pd.DataFrame:
        r"""
        The main effects per observation and featuren (i.e., the diagonals of the
        interaction matrices), with shape
        :math:`(n_\mathrm{observations}, n_\mathrm{outputs} \cdot n_\mathrm{features})`.

        :raise AttributeError: this SHAP calculator does not support interaction values
        """

        if not self.interaction_values:
            raise AttributeError("main effects are only defined for interaction values")

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

    def validate_features(self, features: pd.DataFrame) -> None:
        """
        Check that the given feature matrix is valid for this calculator.

        :param features: the feature matrix to validate
        :raise ValueError: if the feature matrix is not compatible with this
            calculator
        """

        features_expected = self.input_names
        if features_expected is None:
            # no input names defined, so we cannot validate the features
            return

        diff = features.columns.symmetric_difference(features_expected)
        if not diff.empty:
            raise ValueError(
                f"Features to be explained do not match the features used to fit the"
                f"learner: expected {features_expected}, got "
                f"{features.columns.tolist()}."
            )

    def _reset_fit(self) -> None:
        # set this calculator to its initial unfitted state
        self.shap_ = None
        self.feature_index_ = None
        self.output_names_ = None

    def _make_explainer(self, features: pd.DataFrame) -> BaseExplainer:
        # prepare the background dataset

        background_dataset: Optional[pd.DataFrame]

        if self.explainer_factory.uses_background_dataset:
            background_dataset = features

            background_dataset_not_na = background_dataset.dropna()

            if len(background_dataset_not_na) != len(background_dataset):
                n_original = len(background_dataset)
                n_dropped = n_original - len(background_dataset_not_na)
                log.warning(
                    f"{n_dropped} out of {n_original} observations in the background "
                    f"dataset have missing values after pre-processing and will be "
                    f"dropped."
                )

                background_dataset = background_dataset_not_na

        else:
            background_dataset = None

        model = self.model
        explainer = self.explainer_factory.make_explainer(
            model=model, data=background_dataset
        )

        if self.n_jobs != 1:
            explainer = ParallelExplainer(
                explainer,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )

        return explainer

    def _calculate_shap(
        self, *, features: pd.DataFrame, explainer: BaseExplainer
    ) -> pd.DataFrame:
        if features.isna().values.any():
            log.warning(
                "preprocessed features passed to SHAP explainer include NaN values; "
                "try to change preprocessing to impute all NaN values"
            )

        multi_output_index_name = self.MULTI_OUTPUT_INDEX_NAME
        multi_output_names = self.output_names
        assert self.feature_index_ is not None, ASSERTION__CALCULATOR_IS_FITTED
        feature_names = self.feature_index_

        # calculate the shap values, and ensure the result is a list of arrays
        shap_values: List[npt.NDArray[np.float_]] = self._convert_shap_tensors_to_list(
            shap_tensors=(
                explainer.shap_interaction_values(X=features)
                if self.interaction_values
                else explainer.shap_values(X=features)
            ),
            n_outputs=len(multi_output_names),
        )

        # convert to a data frame per output (different logic depending on whether
        # we have a regressor or a classifier, implemented by method
        # shap_matrix_for_split_to_df_fn)

        shap_values_df_per_output: List[pd.DataFrame] = self._convert_shap_to_df(
            raw_shap_tensors=shap_values,
            observation_idx=features.index,
            feature_idx=feature_names,
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
                names=[multi_output_index_name, feature_names.name],
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

        if isinstance(shap_tensors, list):
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

    @abstractmethod
    def _convert_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observation_idx: pd.Index,
        feature_idx: pd.Index,
    ) -> List[pd.DataFrame]:
        """
        Convert the SHAP tensors for a single split to a data frame.

        :param raw_shap_tensors: the raw values returned by the SHAP explainer
        :param observation_idx: the ids used for indexing the explained observations
        :param feature_idx: the feature names
        :return: SHAP values of a single split as data frame
        """
        pass

    def _convert_raw_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observation_idx: pd.Index,
        feature_idx: pd.Index,
    ) -> List[pd.DataFrame]:
        # Convert "raw output" shap tensors to data frames.
        # This is typically the output obtained for regressors, or generic functions.
        if self.interaction_values:
            row_index = pd.MultiIndex.from_product(
                iterables=(observation_idx, feature_idx),
                names=(observation_idx.name, feature_idx.name),
            )

            return [
                pd.DataFrame(
                    data=raw_interaction_tensor.reshape(
                        (-1, raw_interaction_tensor.shape[2])
                    ),
                    index=row_index,
                    columns=feature_idx,
                )
                for raw_interaction_tensor in raw_shap_tensors
            ]
        else:
            return [
                pd.DataFrame(
                    data=raw_shap_matrix, index=observation_idx, columns=feature_idx
                )
                for raw_shap_matrix in raw_shap_tensors
            ]


__tracker.validate()
