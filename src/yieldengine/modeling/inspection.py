from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator

from yieldengine.loading.sample import Sample
from yieldengine.modeling.factory import ModelPipelineFactory


class ModelInspector:
    __slots__ = [
        "_estimator",
        "_pipeline_factory",
        "_cv",
        "_sample",
        "_shap_explainer",
        "_estimators_by_fold",
    ]

    F_FOLD_START = "fold_start"
    F_PREDICTION = "prediction"

    def __init__(
        self,
        estimator: BaseEstimator,
        pipeline_factory: ModelPipelineFactory,
        cv: BaseCrossValidator,
        sample: Sample,
    ) -> None:

        self._estimator = estimator
        self._pipeline_factory = pipeline_factory
        self._cv = cv
        self._sample = sample
        self._estimators_by_fold: Dict[int, BaseEstimator] = {}

        # init the Shap explainer:
        # self._shap_explainer = shap.TreeExplainer(estimator) or other (is Shap
        # able to
        # determine the best explainer based on model type?)

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
            test_sample = self.sample.select_observations(indices=test_indices)
            fold = test_indices[0]

            self._estimators_by_fold[fold] = estimator = clone(self._estimator)

            return pd.DataFrame(
                data={
                    self.F_FOLD_START: fold,
                    self.F_PREDICTION: self._pipeline_factory.make_pipeline(estimator)
                    .fit(X=train_sample.features, y=train_sample.target)
                    .predict(X=test_sample.features),
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
        if len(self._estimators_by_fold) == 0:
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
