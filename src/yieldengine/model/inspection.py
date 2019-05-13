import logging
from typing import *

import numpy as np
import pandas as pd
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator

from yieldengine.loading.sample import Sample
from scipy.stats import pearsonr
from sklearn.cluster import AgglomerativeClustering

log = logging.getLogger(__name__)


class ModelInspector:
    __slots__ = [
        "_estimator",
        "_cv",
        "_sample",
        "_shap_explainer_by_fold",
        "_estimators_by_fold",
        "_predictions_for_all_samples",
        "_shap_matrix",
        "_shap_correlation_matrix",
    ]

    F_FOLD_ID = "fold_start"
    F_PREDICTION = "prediction"
    F_FEATURE_1 = "feature_1"
    F_CLUSTER_LABEL = "cluster_label"

    def __init__(
        self, estimator: BaseEstimator, cv: BaseCrossValidator, sample: Sample
    ) -> None:

        self._estimator = estimator
        self._cv = cv
        self._sample = sample
        self._estimators_by_fold: Union[Dict[int, BaseEstimator], None] = None
        self._predictions_for_all_samples: Union[pd.DataFrame, None] = None
        self._shap_matrix: Union[pd.DataFrame, None] = None
        self._shap_correlation_matrix: Union[pd.DataFrame, None] = None

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
        if self._estimators_by_fold is None:
            self.predictions_for_all_samples()
        return self._estimators_by_fold[fold]

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

        self._estimators_by_fold: Dict[int, BaseEstimator] = {}

        sample = self.sample

        def predict(
            fold_id: int, train_indices: np.ndarray, test_indices: np.ndarray
        ) -> pd.DataFrame:
            train_sample = sample.select_observations(indices=train_indices)
            test_sample = sample.select_observations(indices=test_indices)

            self._estimators_by_fold[fold_id] = estimator = clone(self._estimator)

            return pd.DataFrame(
                data={
                    self.F_FOLD_ID: fold_id,
                    self.F_PREDICTION: estimator.fit(
                        X=train_sample.features, y=train_sample.target
                    ).predict(X=test_sample.features),
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

        def shap_matrix_for_fold(
            fold_id: int, estimator: BaseEstimator
        ) -> pd.DataFrame:
            observation_indices_in_fold = predictions_by_observation_and_fold.xs(
                key=fold_id, level=self.F_FOLD_ID
            ).index

            fold_x = sample.select_observations(
                indices=observation_indices_in_fold
            ).features

            shap_matrix = ModelInspector._make_shap_explainer(
                estimator=estimator, data=fold_x
            ).shap_values(fold_x)

            return pd.DataFrame(
                data=shap_matrix,
                index=observation_indices_in_fold,
                columns=sample.feature_names,
            )

        shap_value_dfs = [
            shap_matrix_for_fold(fold_id, estimator)
            for fold_id, estimator in self._estimators_by_fold.items()
        ]

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = (
            pd.concat(objs=shap_value_dfs)
            .groupby(by=pd.concat(objs=shap_value_dfs).index)
            .mean()
        )

        return self._shap_matrix

    def feature_dependencies(self) -> pd.DataFrame:
        if self._shap_correlation_matrix is not None:
            return self._shap_correlation_matrix

        data = []
        shap_matrix = self.shap_matrix()

        for c1 in shap_matrix.columns:
            row = {
                # pearsonr() returns a tuple of (r, p-value) -> retrieve only r using[0]
                c2: pearsonr(x=shap_matrix.loc[:, c1], y=shap_matrix.loc[:, c2])[0]
                for c2 in shap_matrix.columns
            }
            row[ModelInspector.F_FEATURE_1] = c1
            data.append(row)

        self._shap_correlation_matrix = pd.DataFrame(data).set_index(
            keys=ModelInspector.F_FEATURE_1
        )

        return self._shap_correlation_matrix

    def clustered_feature_dependencies(self, n_clusters: int = 10) -> pd.DataFrame:
        clustering = AgglomerativeClustering(
            linkage="single", affinity="precomputed", n_clusters=n_clusters
        )
        feature_dependencies = self.feature_dependencies()

        # todo: analyse+handle NaN columns better...
        cluster_labels: np.ndarray = clustering.fit_predict(
            X=feature_dependencies.fillna(0).values
        )

        feature_dependencies[ModelInspector.F_CLUSTER_LABEL] = cluster_labels

        return feature_dependencies
