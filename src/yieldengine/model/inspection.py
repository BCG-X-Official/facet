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
    F_FEATURE = "feature"
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

        shap_matrix = self.shap_matrix()

        # collect row-wise in this list in order to convert to pd.DF later
        data = []

        # construct an correlation matrix that looks like this:
        #
        #                    F1         F2         F3         FN
        # feature
        # F1                  1       corr       corr       corr
        # F2               corr          1       corr       corr
        # F3               corr       corr          1       corr
        # FN               corr       corr       corr          1
        #
        for column1 in shap_matrix.columns:
            row = {
                column2: 0
                # for features, for which across all predictions their shap-
                # value is 0, the correlation does not exist, can not be computed
                # and will yield a na (due division by zero)!
                # hence set a correlation of 0 -> "no correlation"
                if (
                    (shap_matrix.loc[:, column1].unique() == 0).all()
                    or (shap_matrix.loc[:, column2].unique() == 0).all()
                )
                # pearsonr() returns a tuple of (r, p-value)
                #   -> retrieve only "r" correlation coefficient using[0]
                else pearsonr(
                    x=shap_matrix.loc[:, column1], y=shap_matrix.loc[:, column2]
                )[0]
                # iterate over all feature columns in the shap matrix
                for column2 in shap_matrix.columns
            }

            # set the index column for this row to the feature column name
            row[ModelInspector.F_FEATURE] = column1

            # append this row
            data.append(row)

        # create a data frame and set the index to "F_FEATURE_1"
        self._shap_correlation_matrix = pd.DataFrame(data).set_index(
            keys=ModelInspector.F_FEATURE
        )

        return self._shap_correlation_matrix

    def clustered_feature_dependencies(
        self, n_clusters: int = 10, remove_all_zero: bool = True
    ) -> pd.DataFrame:
        # set up the clustering estimator
        clustering = AgglomerativeClustering(
            linkage="single", affinity="precomputed", n_clusters=n_clusters
        )

        # retrieve feature_dependencies - already in the correct shape
        feature_dependencies = self.feature_dependencies()

        if remove_all_zero:
            to_drop = set()

            for column in feature_dependencies.columns:
                if (feature_dependencies.loc[:, column].unique() == 0).all():
                    to_drop.add(column)

            feature_dependencies = feature_dependencies.drop(labels=to_drop, axis=1)
            feature_dependencies = feature_dependencies.drop(labels=to_drop, axis=0)

            log.info(f"removed {len(to_drop)} features with all-zero correlation")

        # fit the clustering algorithm using the feature_dependencies as a
        # distance matrix
        # todo: double check if the shift is needed
        # we shift the [-1, 1] correlation coefficient by -1 into [-2, 0],
        # and take the absolute.
        # this means, r = -1 gets a distance of 2 and r = 1 a distance of 0, and so on
        # and then fit the clustering algorithm:
        cluster_labels: np.ndarray = clustering.fit_predict(
            X=np.abs(feature_dependencies.values - 1)
        )

        # return a data frame with the cluster labels added as a series
        feature_dependencies[ModelInspector.F_CLUSTER_LABEL] = cluster_labels

        return feature_dependencies
