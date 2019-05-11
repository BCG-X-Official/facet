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
    ]

    F_FOLD_START = "fold_start"
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
        self._estimators_by_fold: Dict[int, BaseEstimator] = {}
        self._shap_explainer_by_fold: Dict[int, Explainer] = {}

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

    def _preprocess_sample(self) -> None:
        if self._sample_preprocessed is None:
            self._sample_preprocessed = self._preprocessor.process(sample=self._sample)

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

        self._preprocess_sample()

        def predict(
            train_indices: np.ndarray, test_indices: np.ndarray
        ) -> pd.DataFrame:
            train_sample = self._sample_preprocessed.select_observations(
                indices=train_indices
            )
            test_sample = self._sample_preprocessed.select_observations(
                indices=test_indices
            )
            fold = test_indices[0]

            self._estimators_by_fold[fold] = estimator = clone(self._estimator)

            return pd.DataFrame(
                data={
                    self.F_FOLD_START: fold,
                    self.F_PREDICTION: estimator.fit(
                        X=train_sample.features, y=train_sample.target
                    ).predict(X=test_sample.features),
                },
                index=test_sample.index,
            )

        return pd.concat(
            [
                predict(train_indices, test_indices)
                for train_indices, test_indices in self.cv.split(
                    self._sample_preprocessed.features, self._sample_preprocessed.target
                )
            ]
        )

    def estimator(self, fold: int) -> BaseEstimator:
        """
        :param fold: start index of test fold
        :return: the estimator that was used to predict the dependent vartiable of
        the test fold
        """
        if len(self._estimators_by_fold) == 0:
            self.predictions_for_all_samples()

        return self._estimators_by_fold[fold]

    def shap_value_matrix(self) -> pd.DataFrame:
        predictions = self.predictions_for_all_samples()

        shap_value_dfs = []

        for fold, estimator in self._estimators_by_fold.items():
            fold_indices = predictions.loc[
                predictions[self.F_FOLD_START] == fold, :
            ].index

            fold_x = self._sample_preprocessed.select_observations(
                indices=fold_indices
            ).features

            explainer = ModelInspector._make_shap_explainer(
                estimator=estimator, data=fold_x
            )
            self._shap_explainer_by_fold[fold] = explainer

            shap_matrix = explainer.shap_values(fold_x)

            fold_shap_values = pd.DataFrame(
                data=shap_matrix,
                index=fold_indices,
                columns=self._sample_preprocessed.feature_names,
            )

            shap_value_dfs.append(fold_shap_values)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        concatenated = pd.concat(objs=shap_value_dfs)
        aggregated = concatenated.groupby(by=concatenated.index).agg("mean")
        return aggregated

    def feature_dependencies(self) -> pd.DataFrame:
        # calculate shap_value_matrix()
        # find correlations
        # return as DataFrame
        pass

    def clustered_feature_importance(self) -> pd.DataFrame:
        # calculate shap_value_matrix()
        # run hierarchichal clustering
        # return clustering result as DataFrame
        pass
