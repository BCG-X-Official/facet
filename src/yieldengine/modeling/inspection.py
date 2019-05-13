import logging
from typing import *

import numpy as np
import pandas as pd
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator

from yieldengine.loading.sample import Sample
from yieldengine.preprocessing import SamplePreprocessor

log = logging.getLogger(__name__)


class ModelInspector:
    __slots__ = [
        "_estimator",
        "_preprocessor",
        "_cv",
        "_sample",
        "_sample_preprocessed",
        "_shap_explainer_by_fold",
        "_estimators_by_fold",
        "_predictions_for_all_samples",
        "_shap_matrix",
    ]

    F_FOLD_ID = "fold_start"
    F_PREDICTION = "prediction"

    def __init__(
        self,
        preprocessor: SamplePreprocessor,
        estimator: BaseEstimator,
        cv: BaseCrossValidator,
        sample: Sample,
    ) -> None:

        self._estimator = estimator
        self._preprocessor = preprocessor
        self._cv = cv
        self._sample = sample
        self._sample_preprocessed: Union[Sample, None] = None
        self._estimators_by_fold: Union[Dict[int, BaseEstimator], None] = None
        self._predictions_for_all_samples: Union[pd.DataFrame, None] = None
        self._shap_matrix: Union[pd.DataFrame, None] = None

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

    @property
    def sample_preprocessed(self) -> Sample:
        if self._sample_preprocessed is None:
            self._sample_preprocessed = self._preprocessor.process(sample=self._sample)
        return self._sample_preprocessed

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
        sample_preprocessed = self.sample_preprocessed

        def predict(
            fold_id: int, train_indices: np.ndarray, test_indices: np.ndarray
        ) -> pd.DataFrame:
            train_sample = sample_preprocessed.select_observations(
                indices=train_indices
            )
            test_sample = sample_preprocessed.select_observations(indices=test_indices)

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
                    self.cv.split(
                        sample_preprocessed.features, sample_preprocessed.target
                    )
                )
            ]
        )

        return self._predictions_for_all_samples

    def shap_matrix(self) -> pd.DataFrame:
        if self._shap_matrix is not None:
            return self._shap_matrix

        predictions_by_observation_and_fold = self.predictions_for_all_samples().set_index(
            self.F_FOLD_ID, append=True
        )
        sample_preprocessed = self.sample_preprocessed

        def shap_matrix_for_fold(
            fold_id: int, estimator: BaseEstimator
        ) -> pd.DataFrame:
            observation_indices_in_fold = predictions_by_observation_and_fold.xs(
                key=fold_id, level=self.F_FOLD_ID
            ).index

            fold_x = sample_preprocessed.select_observations(
                indices=observation_indices_in_fold
            ).features

            shap_matrix = ModelInspector._make_shap_explainer(
                estimator=estimator, data=fold_x
            ).shap_values(fold_x)

            return pd.DataFrame(
                data=shap_matrix,
                index=observation_indices_in_fold,
                columns=sample_preprocessed.feature_names,
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
        # calculate shap_matrix()
        # find correlations
        # return as DataFrame
        pass

    def clustered_feature_importance(self) -> pd.DataFrame:
        # calculate shap_matrix()
        # run hierarchichal clustering
        # return clustering result as DataFrame
        pass
