from typing import *

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline

from yieldengine.loading.sample import Sample


class ModelInspector:
    __slots__ = ["_pipeline", "_cv", "_sample", "_shap_explainer"]

    _estimators_by_fold: Dict[int, BaseEstimator]

    F_OBSERVATION_ID = "observation"
    F_FOLD_START = "fold_start"
    F_PREDICTION = "prediction"

    def __init__(
        self, pipeline: Pipeline, cv: BaseCrossValidator, sample: Sample
    ) -> None:
        if not isinstance(
            ModelInspector._extract_estimator_from_pipeline(pipeline), BaseEstimator
        ):
            raise ValueError("arg pipeline does not end with estimator")
        self._pipeline = pipeline
        self._cv = cv
        self._sample = sample

        # init the Shap explainer:
        # self._shap_explainer = shap.TreeExplainer(estimator) or other (is Shap
        # able to
        # determine the best explainer based on model type?)

    @staticmethod
    def _extract_estimator_from_pipeline(pipeline: Pipeline) -> BaseEstimator:
        return pipeline.steps[-1][1]

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def cv(self) -> BaseCrossValidator:
        return self._cv

    @property
    def sample(self) -> Sample:
        return self._sample

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

        def predict(
            train_indices: np.ndarray, test_indices: np.ndarray
        ) -> pd.DataFrame:
            train_sample = self.sample.select_observations(indices=train_indices)
            test_sample = self.sample.select_observations(indices=train_indices)
            fold = test_indices[0]
            self._estimators_by_fold[fold] = clone(
                self._extract_estimator_from_pipeline(
                    self.pipeline.fit(train_sample.features, train_sample.target)
                )
            )
            return pd.DataFrame(
                data={
                    self.F_FOLD_START: fold,
                    self.F_PREDICTION: self.pipeline.predict(X=test_sample.features),
                },
                index=test_sample.index,
            )

        return pd.concat(
            [
                predict(train_indices, test_indices)
                for train_indices, test_indices in self.cv.split(
                    self.sample.features, self.sample.target
                )
            ]
        )

    def get_estimator(self, fold: int) -> BaseEstimator:
        """
        :param fold: start index of test fold
        :return: the estimator that was used to predict the dependent vartiable of
        the test fold
        """
        if self._estimators_by_fold is None:
            self.predictions_for_all_samples()

        return self._estimators_by_fold[fold]

    def shap_value_matrix(self) -> pd.DataFrame:
        pass

        # 1.) Run the predictor with the sample; get resulting predictions DF

        # 2.) Loop over predictions, explain each  & build resulting SHAP matrix

        # 3.) Group SHAP matrix by observation ID and aggregate SHAP values using mean()

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
