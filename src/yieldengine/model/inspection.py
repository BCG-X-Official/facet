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
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from yieldengine import deprecated, Sample
from yieldengine.dendrogram import LinkageTree
from yieldengine.model import Model

log = logging.getLogger(__name__)


class ModelInspector:
    __slots__ = [
        "_model",
        "_cv",
        "_sample",
        "_shap_explainer_by_fold",
        "_predictions_for_all_samples",
        "_shap_matrix",
        "_feature_dependency_matrix",
        "_model_by_fold",
    ]

    F_FOLD_ID = "fold_id"
    F_PREDICTION = "prediction"
    F_FEATURE = "feature"
    F_CLUSTER_LABEL = "cluster_label"

    def __init__(self, model: Model, cv: BaseCrossValidator, sample: Sample) -> None:

        self._model = model
        self._cv = cv
        self._sample = sample
        self._model_by_fold: Optional[Dict[int, Model]] = None
        self._predictions_for_all_samples: Optional[pd.DataFrame] = None
        self._shap_matrix: Optional[pd.DataFrame] = None
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None

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
            return TreeExplainer(
                model=estimator, data=data, feature_dependence="independent"
            )
        except Exception as e:
            log.debug(
                f"failed to instantiate shap.TreeExplainer:{str(e)},"
                "using shap.KernelExplainer as fallback"
            )
            # when using KernelExplainer, shap expects "model" to be a callable that
            # predicts
            # noinspection PyUnresolvedReferences
            return KernelExplainer(model=estimator.predict, data=data)

    @property
    def cv(self) -> BaseCrossValidator:
        return self._cv

    @property
    def sample(self) -> Sample:
        return self._sample

    def estimator(self, fold: int) -> BaseEstimator:
        """
        :param fold: start index of test fold
        :return: the estimator that was used to predict the dependent variable of
        the test fold
        """
        if self._model_by_fold is None:
            self.predictions_for_all_samples()
        return self._model_by_fold[fold].estimator

    def predictions_for_all_samples(self) -> pd.DataFrame:
        """
        For each fold of this Predictor's CV, fit the estimator and predict all
        values in the test set. The result is a data frame with one row per
        prediction, indexed by the observations in the sample, and with columns
        F_FOLD_START (the numerical index of the start of the test set in the current
        fold), F_PREDICTION (the predicted value for the given observation and fold)

        Note that there can be multiple prediction rows per observation if the test
        folds overlap.

        :return: the data frame with the predictions per observation and test fold
        """

        # 1. execute the preprocessing pipeline on the sample
        # 2. get fold splits across the preprocessed observations using self._cv
        # 3. for each split
        #       - clone the estimator using function sklearn.base.clone()
        #       - fit the estimator on the X and y of train set
        #       - predict y for test set

        if self._predictions_for_all_samples is not None:
            return self._predictions_for_all_samples

        self._model_by_fold: Dict[int, Pipeline] = {}

        sample = self.sample

        def predict(
            fold_id: int, train_indices: np.ndarray, test_indices: np.ndarray
        ) -> pd.DataFrame:
            train_sample = sample.select_observations(indices=train_indices)
            test_sample = sample.select_observations(indices=test_indices)

            self._model_by_fold[fold_id] = model = self._model.clone()

            pipeline = model.pipeline()

            pipeline.fit(X=train_sample.features, y=train_sample.target)

            return pd.DataFrame(
                data={
                    self.F_FOLD_ID: fold_id,
                    self.F_PREDICTION: pipeline.predict(X=test_sample.features),
                },
                index=test_sample.index,
            )

        self._predictions_for_all_samples = pd.concat(
            [
                predict(fold_id, train_indices, test_indices)
                for fold_id, (train_indices, test_indices) in enumerate(
                    self.cv.split(sample.features, sample.target)
                )
            ]
        )

        return self._predictions_for_all_samples

    def shap_matrix(self) -> pd.DataFrame:
        if self._shap_matrix is not None:
            return self._shap_matrix

        sample = self.sample

        predictions_by_observation_and_fold = self.predictions_for_all_samples().set_index(
            self.F_FOLD_ID, append=True
        )

        def shap_matrix_for_fold(fold_id: int, fold_model: Model) -> pd.DataFrame:

            observation_indices_in_fold = predictions_by_observation_and_fold.xs(
                key=fold_id, level=self.F_FOLD_ID
            ).index

            fold_x = sample.select_observations(
                indices=observation_indices_in_fold
            ).features

            estimator = fold_model.estimator

            if fold_model.preprocessing is not None:
                data_transformed = fold_model.preprocessing.transform(fold_x)
            else:
                data_transformed = fold_x

            shap_matrix = ModelInspector._make_shap_explainer(
                estimator=estimator, data=data_transformed
            ).shap_values(data_transformed)

            return pd.DataFrame(
                data=shap_matrix,
                index=observation_indices_in_fold,
                columns=data_transformed.columns,
            )

        shap_value_dfs = [
            shap_matrix_for_fold(fold_id, model)
            for fold_id, model in self._model_by_fold.items()
        ]

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = (
            pd.concat(objs=shap_value_dfs, sort=True)
            .groupby(by=pd.concat(objs=shap_value_dfs).index, sort=True)
            .mean()
        )

        return self._shap_matrix

    def feature_importances(self) -> pd.Series:
        """
        :returns feature importances as their mean absolute SHAP contributions,
        normalised to a total 100%
        """
        feature_importances: pd.Series = self.shap_matrix().abs().mean()
        return feature_importances / feature_importances.sum()

    def feature_dependency_matrix(self) -> pd.DataFrame:
        if self._feature_dependency_matrix is None:
            shap_matrix = self.shap_matrix()

            # exclude features with zero Shapley importance
            shap_matrix = shap_matrix.loc[:, shap_matrix.sum() > 0]

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
