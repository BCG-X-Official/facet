# coding=utf-8
"""
Inspection of a model.

The :class:`ModelInspector` class computes the shap matrix and the associated linkage
tree of a model which has been fitted using cross-validation.
"""
import logging
from typing import *

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer

from yieldengine.model import ModelPipelineDF
from yieldengine.model.prediction import PredictorFitCV
from yieldengine.sklearndf import DataFramePredictor
from yieldengine.sklearndf._wrapper import DataFrameEstimatorWrapper
from yieldengine.viz.dendrogram import LinkageTree

log = logging.getLogger(__name__)


class ModelInspector:
    """
    Inspect a model through its shap values.

    :param predictor_fit: predictor containing the information about the
      model, the data (a Sample object), the cross-validation and predictions.
    :param explainer_factory: method that returns a shap Explainer
    """

    __slots__ = [
        "_shap_matrix",
        "_feature_dependency_matrix",
        "_predictor_fit",
        "_explainer_factory",
    ]

    F_FEATURE = "feature"

    def __init__(
        self,
        predictor_fit: PredictorFitCV,
        explainer_factory: Optional[
            Callable[[DataFramePredictor, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:

        self._shap_matrix: Optional[pd.DataFrame] = None
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None
        self._predictor_fit = predictor_fit
        self._explainer_factory = (
            explainer_factory
            if explainer_factory is not None
            else default_explainer_factory
        )

    @property
    def model_fit(self) -> PredictorFitCV:
        """:class:`PredictorFitCV` used for inspection."""
        return self._predictor_fit

    def shap_matrix(self) -> pd.DataFrame:
        """
        Calculate the SHAP matrix for all splits.

        Each row is an observation in a specific test split, and each column is a
        feature, and values are the SHAP values per observation/split and feature.

        :return: shap matrix as a data frame
        """
        if self._shap_matrix is not None:
            return self._shap_matrix

        sample = self.model_fit.sample

        predictions_by_observation_and_split = (
            self.model_fit.predictions_for_all_splits()
        )

        def _shap_matrix_for_split(
            split_id: int, split_model: ModelPipelineDF
        ) -> pd.DataFrame:

            observation_indices_in_split = predictions_by_observation_and_split.xs(
                key=split_id, level=PredictorFitCV.F_SPLIT_ID
            ).index

            split_x = sample.select_observations_by_index(
                ids=observation_indices_in_split
            ).features

            estimator = split_model.predictor
            if isinstance(estimator, DataFrameEstimatorWrapper):
                estimator = estimator.base_estimator

            if split_model.preprocessing is not None:
                data_transformed = split_model.preprocessing.transform(split_x)
            else:
                data_transformed = split_x

            shap_matrix = self._explainer_factory(
                estimator=estimator, data=data_transformed
            ).shap_values(data_transformed)

            # todo: we need another condition to handle LGBM's (inconsistent) output
            if isinstance(shap_matrix, list):
                # the shap explainer returns an array [obs x features] for each of the
                # target-classes

                n_arrays = len(shap_matrix)

                # we decided to support only binary classification == 2 classes:
                assert n_arrays == 2, (
                    f"Expected 2 arrays of shap values in binary classification, "
                    f"got {n_arrays}"
                )

                # in the binary classification case, we will proceed with shap values
                # for class 0, since values for class 1 will just be the same
                # values times (*-1)  (the opposite probability)

                # to assure the values are returned as expected above,
                # and no information of class 1 is discarded, assert the
                # following:
                assert np.all(
                    (shap_matrix[0]) - (shap_matrix[1] * -1) < 1e-10
                ), "Expected shap_matrix(class 0) == shap_matrix(class 1) * -1"

                shap_matrix = shap_matrix[0]

            if not isinstance(shap_matrix, np.ndarray):
                log.warning(
                    f"shap explainer output expected to be an ndarray but was "
                    f"{type(shap_matrix)}"
                )

            return pd.DataFrame(
                data=shap_matrix,
                index=observation_indices_in_split,
                columns=data_transformed.columns,
            )

        shap_values_df = pd.concat(
            objs=[
                _shap_matrix_for_split(split_id, model)
                for split_id, model in enumerate(self.model_fit.fitted_models())
            ],
            sort=True,
        ).fillna(0.0)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = shap_values_df.groupby(by=shap_values_df.index).mean()

        return self._shap_matrix

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


def default_explainer_factory(
    estimator: DataFramePredictor, data: pd.DataFrame
) -> Explainer:
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

    # further there is no consistent "ModelPipelineDF type X is unsupported" exception raised,
    # which is why we need to always assume the error resulted from this cause -
    # we should not attempt to filter the exception type or message given that it is
    # currently inconsistent

    if isinstance(estimator, DataFrameEstimatorWrapper):
        estimator = estimator.base_estimator

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
