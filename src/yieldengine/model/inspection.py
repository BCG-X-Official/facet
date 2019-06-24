import logging
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from yieldengine import deprecated
from yieldengine.dendrogram import LinkageTree
from yieldengine.model import Model
from yieldengine.model.prediction import PredictorCV

log = logging.getLogger(__name__)


class ModelInspector:
    __slots__ = [
        "_shap_explainer_by_split",
        "_shap_matrix",
        "_feature_dependency_matrix",
        "_predictor",
    ]

    F_FEATURE = "feature"

    def __init__(self, predictor: PredictorCV) -> None:

        self._shap_matrix: Optional[pd.DataFrame] = None
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None
        self._predictor = predictor

    @property
    def predictor(self) -> PredictorCV:
        return self._predictor

    @staticmethod
    def _make_shap_explainer(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:

        # NOTE:
        # unfortunately, there is no convenient function in shap to determine the best
        # explainer method. hence we use this try/except approach.

        # further there is no consistent "Modeltype X is unsupported" exception raised,
        # which is why we need to always assume the error resulted from this cause -
        # we should not attempt to filter the exception type or message given that it is
        # currently inconsistent

        # todo: instead create factory for shap explainers
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

    def shap_matrix(self) -> pd.DataFrame:
        if self._shap_matrix is not None:
            return self._shap_matrix

        sample = self.predictor.sample

        predictions_by_observation_and_split = self.predictor.predictions_for_all_samples().set_index(
            PredictorCV.F_SPLIT_ID, append=True
        )

        def shap_matrix_for_split(split_id: int, split_model: Model) -> pd.DataFrame:

            observation_indices_in_split = predictions_by_observation_and_split.xs(
                key=split_id, level=PredictorCV.F_SPLIT_ID
            ).index

            split_x = sample.select_observations(
                ids=observation_indices_in_split
            ).features

            estimator = split_model.estimator

            if split_model.preprocessing is not None:
                data_transformed = split_model.preprocessing.transform(split_x)
            else:
                data_transformed = split_x

            shap_matrix = ModelInspector._make_shap_explainer(
                estimator=estimator, data=data_transformed
            ).shap_values(data_transformed)

            return pd.DataFrame(
                data=shap_matrix,
                index=observation_indices_in_split,
                columns=data_transformed.columns,
            )

        shap_values_df = pd.concat(
            objs=[
                shap_matrix_for_split(split_id, model)
                for split_id, model in self.predictor.model_by_split.items()
            ],
            sort=True,
        ).fillna(0.0)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = shap_values_df.groupby(by=shap_values_df.index).mean()

        return self._shap_matrix

    def feature_importances(self) -> pd.Series:
        """
        :returns feature importances as their mean absolute SHAP contributions,
        normalised to a total 100%
        """
        feature_importances: pd.Series = self.shap_matrix().abs().mean()
        return (feature_importances / feature_importances.sum()).sort_values(
            ascending=False
        )

    def feature_dependency_matrix(self) -> pd.DataFrame:
        if self._feature_dependency_matrix is None:
            shap_matrix = self.shap_matrix()

            # exclude features with zero Shapley importance
            # noinspection PyUnresolvedReferences
            shap_matrix = shap_matrix.loc[:, (shap_matrix != 0.0).any()]

            self._feature_dependency_matrix = shap_matrix.corr(method="pearson")

        return self._feature_dependency_matrix

    def cluster_dependent_features(self) -> LinkageTree:
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

    @deprecated("experimental version using scipy natively")
    def plot_feature_dendrogram_scipy(
        self, figsize: Tuple[int, int] = (20, 30)
    ) -> None:

        feature_dependencies = self.feature_dependency_matrix()

        compressed = squareform(1 - np.abs(feature_dependencies.values))

        linkage_matrix = linkage(y=compressed, method="single")
        pyplot.figure(num=None, figsize=figsize, facecolor="w", edgecolor="k")
        dendrogram(
            linkage_matrix,
            labels=feature_dependencies.index.values,
            orientation="left",
            distance_sort=True,
            leaf_font_size=16,
        )
        pyplot.axvline(x=0.5, ymin=0, ymax=1, linestyle="--", color="#000000")
        pyplot.axvline(x=0.25, ymin=0, ymax=1, linestyle="--", color="#000000")
        pyplot.title("Hierarchical Clustering: Correlated Feature Dependence")

        pyplot.show()
