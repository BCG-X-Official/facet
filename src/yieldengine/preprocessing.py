from abc import ABC, abstractmethod
from typing import *

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from yieldengine import Sample


class SamplePreprocessor(ABC):

    __slots__ = ["_preprocessing_transformer"]

    def __init__(self, preprocessing_transformer: ColumnTransformer):
        self._preprocessing_transformer = preprocessing_transformer

    @abstractmethod
    def process(self, sample: Sample) -> Sample:
        pass

    @property
    def preprocessing_transformer(self) -> ColumnTransformer:
        return self._preprocessing_transformer


class SimpleSamplePreprocessor(SamplePreprocessor):

    STEP_IMPUTE = "impute"
    STEP_ONE_HOT_ENCODE = "one-hot-encode"

    def __init__(
        self, impute_mean: Iterable[str] = None, one_hot_encode: Iterable[str] = None
    ):
        """

        :param impute_mean: list of columns to impute or None
        :param one_hot_encode: list of (categorical) columns to encode or None
        """

        transformations: List[Tuple[str, TransformerMixin, Iterable[str]]] = list()

        if impute_mean is not None and sum(1 for col in impute_mean) > 0:
            transformations.append(
                (
                    SimpleSamplePreprocessor.STEP_IMPUTE,
                    SimpleImputer(strategy="mean"),
                    impute_mean,
                )
            )

        if one_hot_encode is not None and sum(1 for col in one_hot_encode) > 0:
            transformations.append(
                (
                    SimpleSamplePreprocessor.STEP_ONE_HOT_ENCODE,
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    one_hot_encode,
                )
            )

        super().__init__(preprocessing_transformer=ColumnTransformer(transformations))

    def process(self, sample: Sample) -> Any:

        features = sample.features

        transformed_x = self.preprocessing_transformer.fit_transform(X=features)

        feature_names = self._post_transform_feature_names(observations=sample)

        # convert to a DF
        new_observations = pd.DataFrame(data=transformed_x, columns=feature_names)

        new_observations.loc[:, sample.target_name] = sample.target

        # return new Sample
        return Sample(observations=new_observations, target_name=sample.target_name)

    def _post_transform_feature_names(self, observations: Sample) -> List[str]:
        feature_names = []
        for sub_transformer in self.preprocessing_transformer.transformers:
            # sub_transformer is a tuple with 3 elements we can deconstruct.
            #   1. transformer name (str),
            #   2. the transformer object (BaseEstimator),
            #   3. the column names (collection[str])
            t_name, t_obj, t_col_names = sub_transformer

            # is this sub_transformer the OneHotEncoder?
            if t_name == SimpleSamplePreprocessor.STEP_ONE_HOT_ENCODE:
                # annotate the type
                t_obj: OneHotEncoder = t_obj
                # fit the transformer
                t_obj.fit(observations.features.loc[:, t_col_names])
                # retrieve the new feature names from it
                feature_names.extend(
                    t_obj.get_feature_names(input_features=t_col_names)
                )
            elif t_name == SimpleSamplePreprocessor.STEP_IMPUTE:
                # this is the SimpleImputer:
                # col_names are all original column names given to this transformer,
                # but minus those columns that are all N/A - (imputer drops those!)
                col_names = (
                    observations.features.loc[:, t_col_names]
                    .dropna(axis=1, how="all")
                    .columns
                )
                feature_names.extend(col_names)

        return feature_names
