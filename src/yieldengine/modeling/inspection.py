import logging
from typing import *

import numpy as np
import pandas as pd
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import OneHotEncoder

from yieldengine.loading.sample import Sample
from yieldengine.modeling.factory import (
    ModelPipelineFactory,
    PreprocessingModelPipelineFactory,
)
from shap import TreeExplainer, KernelExplainer

log = logging.getLogger(__name__)


class ModelInspector:
    __slots__ = [
        "_estimator",
        "_pipeline_factory",
        "_cv",
        "_sample",
        "_shap_explainer_by_fold",
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
        self._shap_explainer_by_fold: Dict[
            int, Union[TreeExplainer, KernelExplainer]
        ] = {}

    @staticmethod
    def _make_shap_explainer(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:

        # NOTE:
        # unfortunately, there is no convenient function in shap to determine the best
        # explainer method. hence we use this try/except approach.

        # further there is no consistent "Modeltype X is unsupported" exception raised,
        # which is why we need to always assume the error resulted from this cause -
        # we should not attempt to filter the exception type or message given that it is
        # currently inconsistent

        try:
            return TreeExplainer(model=estimator, data=data)
        except Exception as e:
            log.debug(
                f"failed to instantiate shap.TreeExplainer:{str(e)},"
                "using shap.KernelExplainer as fallback"
            )
            # when using KernelExplainer, shap expects "model" to be a callable that
            # predicts
            return KernelExplainer(model=estimator.predict, data=data)

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

    def estimator(self, fold: int) -> BaseEstimator:
        """
        :param fold: start index of test fold
        :return: the estimator that was used to predict the dependent vartiable of
        the test fold
        """
        if len(self._estimators_by_fold) == 0:
            self.predictions_for_all_samples()

        return self._estimators_by_fold[fold]

    def _feature_names(self, X: pd.DataFrame) -> Collection[str]:
        # if there is no preprocessing, we expect unchanged feature names
        if not isinstance(self._pipeline_factory, PreprocessingModelPipelineFactory):
            return self.sample.feature_names

        def extract_encoded_feature_names(
            encoder: OneHotEncoder, X: pd.DataFrame
        ) -> Collection[str]:
            encoder.fit(X=X)
            return encoder.get_feature_names()

        feature_names = []
        # if there is preprocessing, we have to check if transformations altered
        # feature columns and get appropriate names
        main_transformer = self._pipeline_factory.preprocessing_transformer

        # if we found a column transformer, we can get transformed feature names
        # in correct order from it
        if isinstance(main_transformer, ColumnTransformer):
            # annotate the type:
            main_transformer: ColumnTransformer = main_transformer

            for sub_transformer in main_transformer.transformers:
                # sub_transformer is a tuple with 3 elements we can deconstruct.
                #   1. transformer name (str),
                #   2. the transformer object (BaseEstimator),
                #   3. the column names (collection[str])
                t_name, t_obj, t_col_names = sub_transformer

                # is this sub_transformer a OneHot encoder?
                # todo: expand this to other transformers that change features
                if isinstance(t_obj, OneHotEncoder):
                    # annotate the type
                    t_obj: OneHotEncoder = t_obj
                    feature_names.extend(
                        extract_encoded_feature_names(encoder=t_obj, X=X)
                    )
                else:
                    # any other transformer? take original column names
                    feature_names.extend(t_col_names)

        elif isinstance(main_transformer, OneHotEncoder):
            main_transformer: OneHotEncoder = main_transformer
            feature_names.extend(
                extract_encoded_feature_names(encoder=main_transformer, X=X)
            )
        else:
            feature_names = self.sample.feature_names
            log.warning(
                msg="Unkown preprocessing transformer type:"
                f"{main_transformer.__class__}, using original feature names"
            )

        return feature_names

    def shap_value_matrix(self) -> pd.DataFrame:
        predictions = self.predictions_for_all_samples()

        shap_value_dfs = []

        for fold, estimator in self._estimators_by_fold.items():
            fold_indices = predictions.loc[
                predictions[self.F_FOLD_START] == fold, :
            ].index

            fold_x = self.sample.select_observations(indices=fold_indices).features

            explainer = ModelInspector._make_shap_explainer(
                estimator=estimator, data=fold_x
            )
            self._shap_explainer_by_fold[fold] = explainer

            shap_matrix = explainer.shap_values(fold_x.values)

            fold_shap_values = pd.DataFrame(data=shap_matrix, index=fold_indices)

            # set correct column names...
            fold_shap_values.columns = self._feature_names(X=fold_x)

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
