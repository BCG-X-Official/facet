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
Inspection of a model.

The :class:`ModelInspector` class computes the shap matrix and the associated linkage
tree of a model which has been fitted using cross-validation.
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from gamma.ml.fitcv import (
    ClassifierFitCV,
    EstimatorFitCV,
    PredictorFitCV,
    RegressorFitCV,
)
from gamma.ml.viz import LinkageTree
from gamma.sklearndf.pipeline import PredictorPipelineDF

log = logging.getLogger(__name__)

__all__ = [
    "ModelInspector",
    "PredictorInspector",
    "ClassifierInspector",
    "RegressorInspector",
]


T_EstimatorFitCV = TypeVar("T_EstimatorFitCV", bound=EstimatorFitCV)


class ModelInspector(Generic[T_EstimatorFitCV]):

    __slots__ = ["_models"]

    def __init__(self, models: T_EstimatorFitCV) -> None:

        self._models = models

    @property
    def models(self) -> T_EstimatorFitCV:
        """collection of CV-fitted models handled by this modelinspector."""
        return self._models


class PredictorInspector(ModelInspector[PredictorFitCV], ABC):
    """
    Inspect a model through its SHAP values.

    :param models: predictor containing the information about the
      model, the data (a Sample object), the cross-validation and predictions.
    :param explainer_factory: method that returns a shap Explainer
    """

    __slots__ = ["_shap_matrix", "_feature_dependency_matrix", "_explainer_factory"]

    F_FEATURE = "feature"

    def __init__(
        self,
        models: PredictorFitCV,
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:

        super().__init__(models)
        self._shap_matrix: Optional[pd.DataFrame] = None
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None
        self._explainer_factory = (
            explainer_factory
            if explainer_factory is not None
            else tree_explainer_factory
        )

    def shap_matrix(self) -> pd.DataFrame:
        """
        Calculate the SHAP matrix for all splits.

        Each row is an observation in a specific test split, and each column is a
        feature, and values are the SHAP values per observation/split and feature.

        :return: shap matrix as a data frame
        """
        if self._shap_matrix is not None:
            return self._shap_matrix

        shap_values_df = pd.concat(
            objs=[
                self._shap_matrix_for_split(split_id, model)
                for split_id, model in enumerate(self.models)
            ],
            sort=True,
        ).fillna(0.0)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = shap_values_df.groupby(by=shap_values_df.index).mean()

        return self._shap_matrix

    def _shap_matrix_for_split(
        self, split_id: int, split_model: PredictorPipelineDF
    ) -> pd.DataFrame:
        """
        Calaculate the SHAP matrix for a single split.

        :param split_id: the numeric split ID (1,2,3...etc.)
        :param split_model: model trained on the split
        :return: SHAP matrix of a single split as dataframe
        """
        observation_indices_in_split = (
            self.models.predictions_for_all_splits()
            .xs(key=split_id, level=PredictorFitCV.F_SPLIT_ID)
            .index
        )

        split_x = self.models.sample.select_observations_by_index(
            ids=observation_indices_in_split
        ).features

        estimator = split_model.final_estimator_

        if split_model.preprocessing is not None:
            data_transformed = split_model.preprocessing.transform(split_x)
        else:
            data_transformed = split_x

        raw_shap_values = self._explainer_factory(
            estimator=estimator.root_estimator, data=data_transformed
        ).shap_values(data_transformed)

        return self._shap_matrix_for_split_to_df(
            raw_shap_values=raw_shap_values, split_transformed=data_transformed
        )

    @abstractmethod
    def _shap_matrix_for_split_to_df(
        self,
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        split_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a dataframe.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param split_transformed: the transformed data the model was trained on
        :return: SHAP matrix of a single split as dataframe
        """
        pass

    def feature_importances(self) -> pd.Series:
        """
        Feature importance computed using absolute value of shap values.

        :return: feature importances as their mean absolute SHAP contributions,
          normalised to a total 100%
        """
        feature_importances: pd.Series = self.shap_matrix().abs().mean()
        return (feature_importances / feature_importances.sum()).sort_values(
            ascending=False
        )

    def feature_dependency_matrix(self) -> pd.DataFrame:
        """
        Return the Pearson correlation matrix of the shap matrix.

        :return: dataframe with column and index given by the feature names,
          and values are the Pearson correlations of the shap values of features
        """
        if self._feature_dependency_matrix is None:
            shap_matrix = self.shap_matrix()

            # exclude features with zero Shapley importance
            # noinspection PyUnresolvedReferences
            shap_matrix = shap_matrix.loc[:, (shap_matrix != 0.0).any()]

            self._feature_dependency_matrix = shap_matrix.corr(method="pearson")

        return self._feature_dependency_matrix

    def cluster_dependent_features(self) -> LinkageTree:
        """
        Return the :class:`.LinkageTree` based on the `feature_dependency_matrix`.

        :return: linkage tree for the shap clustering dendrogram
        """
        # convert shap correlations to distances (1 = most distant)
        feature_distance_matrix = 1 - self.feature_dependency_matrix().abs()

        # compress the distance matrix (required by scipy)
        compressed_distance_vector = squareform(feature_distance_matrix)

        # calculate the linkage matrix
        linkage_matrix = linkage(y=compressed_distance_vector, method="single")

        # feature labels and weights will be used as the leaves of the linkage tree
        feature_importances = self.feature_importances()

        # select only the features that appear in the distance matrix, and in the
        # correct order
        feature_importances = feature_importances.loc[
            feature_importances.index.intersection(feature_distance_matrix.index)
        ]

        # build and return the linkage tree
        return LinkageTree(
            scipy_linkage_matrix=linkage_matrix,
            leaf_labels=feature_importances.index,
            leaf_weights=feature_importances.values,
        )


def tree_explainer_factory(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.
    Otherwise return :class:`shap.KernelExplainer` which is expected to be much slower.

    :param estimator: estimator from which we want to compute shap values
    :param data: data used to compute the shap values
    :return: :class:`shap.TreeExplainer` if the estimator is compatible,
        else :class:`shap.KernelExplainer`."""

    # NOTE:
    # unfortunately, there is no convenient function in shap to determine the best
    # explainer method. hence we use this try/except approach.
    # further there is no consistent "ModelPipelineDF type X is unsupported"
    # exception raised,
    # which is why we need to always assume the error resulted from this cause -
    # we should not attempt to filter the exception type or message given that it is
    # currently inconsistent

    try:
        return TreeExplainer(model=estimator)
    except Exception as e:
        log.debug(
            f"failed to instantiate shap.TreeExplainer:{str(e)},"
            "using shap.KernelExplainer as fallback"
        )
        # when using KernelExplainer, shap expects "model" to be a callable that
        # predicts
        # noinspection PyUnresolvedReferences
        return KernelExplainer(model=estimator.predict, data=data)


class RegressorInspector(PredictorInspector):
    """
    Inspect a regression model through its SHAP values.

    :param models: regressor containing the information about the
      model, the data (a Sample object), the cross-validation and predictions.
    :param explainer_factory: method that returns a shap Explainer
    """

    def __init__(
        self,
        models: RegressorFitCV,
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:

        super().__init__(models, explainer_factory)

    def _shap_matrix_for_split_to_df(
        self,
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        split_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a dataframe.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param split_transformed: the transformed data the model was trained on
        :return: SHAP matrix of a single split as dataframe
        """

        # In the regression case, only ndarray outputs are expected from SHAP:
        if not isinstance(raw_shap_values, np.ndarray):
            raise ValueError(
                f"shap explainer output expected to be an ndarray but was "
                f"{type(raw_shap_values)}"
            )

        return pd.DataFrame(
            data=raw_shap_values,
            index=split_transformed.index,
            columns=split_transformed.columns,
        )


class ClassifierInspector(PredictorInspector):
    """
    Inspect a classification model through its SHAP values.

    Currently only binary, single-output classification problems are supported.

    :param models: classifier containing the information about the
      model, the data (a Sample object), the cross-validation and predictions.
    :param explainer_factory: method that returns a shap Explainer
    """

    def __init__(
        self,
        models: ClassifierFitCV,
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:

        super().__init__(models, explainer_factory)

    def _shap_matrix_for_split_to_df(
        self,
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        split_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a dataframe.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param split_transformed: the transformed data the model was trained on
        :return: SHAP matrix of a single split as dataframe
        """

        # todo: adapt this method (and override others) to support non-binary
        #   classification

        if isinstance(raw_shap_values, list):
            # the shap explainer returned an array [obs x features] for each of the
            # target-classes

            n_arrays = len(raw_shap_values)

            # we decided to support only binary classification == 2 classes:
            assert n_arrays == 2, (
                "classification model inspection only supports binary classifiers, "
                f"but SHAP analysis returned values for {n_arrays} classes"
            )

            # in the binary classification case, we will proceed with SHAP values
            # for class 0, since values for class 1 will just be the same
            # values times (*-1)  (the opposite probability)

            # to assure the values are returned as expected above,
            # and no information of class 1 is discarded, assert the
            # following:
            assert np.all(
                (raw_shap_values[0]) - (raw_shap_values[1] * -1) < 1e-10
            ), "Expected shap_values(class 0) == shap_values(class 1) * -1"

            # all good: proceed with SHAP values for class 0:
            raw_shap_values = raw_shap_values[0]

        # after the above transformation, `raw_shap_values` should be ndarray:
        if not isinstance(raw_shap_values, np.ndarray):
            raise ValueError(
                f"shap explainer output expected to be an ndarray but was "
                f"{type(raw_shap_values)}"
            )

        return pd.DataFrame(
            data=raw_shap_values,
            index=split_transformed.index,
            columns=split_transformed.columns,
        )
