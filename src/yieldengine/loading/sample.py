import pandas as pd
from typing import List
import numpy as np


class Sample:
    """
    Utility class to wrap a Pandas DataFrame in order to easily access its
        - feature (a DataFrame without the target column) or target (a Series) column(s)
        - feature columns by type: numerical or categorical

    via object properties.

    Further, the features property can be changed.

    An added benefit is through several checks:
        - features & target columns need to be defined explicitly at Sample().__init__() time
        - target column is not allowed as part of the features

    """
    def __init__(
        self, sample: pd.DataFrame, features: List[str] = None, target: str = None
    ) -> None:
        """
        Construct a Sample object.
        :param sample: a Pandas DataFrame
        :param features: list of column names that constitute as feature variables
        :param target: string of column name that constitutes as the target variable
        """
        if sample is None or not (type(sample) == pd.DataFrame):
            raise ValueError("Expected 'sample' to be a pd.DataFrame")

        if features is None or target is None:
            raise ValueError("Both features and target need to be specified")

        self.__features = features
        self.__target = target
        self.__sample = sample

        # finally check values of target & features against sample
        self.__validate_features(self.__features)
        self.__validate_target(self.__target)

        self.__target_data = self.__sample[self.__target]
        self.__feature_data = self.__sample[self.__features]

    def __validate_features(self, features: List[str]) -> None:
        if not set(features).issubset(self.__sample.columns):
            missing_columns = set(features).difference(self.__sample.columns)
            raise ValueError(
                f"Some given features are not in the sample: {missing_columns}"
            )
        # assure target column is not part of features:
        if self.__target in features:
            raise ValueError(
                f"The target column {self.__target} is also part of features"
            )

    def __validate_target(self, target: str) -> None:
        if target not in self.__sample.columns:
            raise ValueError(
                f"The given/assumed target variable {target} is not in the sample"
            )

    @property
    def target(self) -> str:
        return self.__target

    @property
    def features(self) -> List[str]:
        return self.__features

    @property
    def target_data(self) -> pd.Series:
        return self.__target_data

    @property
    def feature_data(self) -> pd.DataFrame:
        return self.__feature_data

    @features.setter
    def features(self, features: List[str]) -> None:
        self.__validate_features(features=features)
        self.__features = features
        self.__feature_data = self.__sample[self.__features]

    @property
    def numerical_features(self) -> List[str]:
        return list(self.__feature_data.select_dtypes(np.number).columns)

    @property
    def categorical_features(self) -> List[str]:
        return list(self.__feature_data.select_dtypes(np.object).columns)

    def __len__(self):
        return len(self.__sample)

    # todo: undecided, if we want to allow the setting of "target" - tendency to not allow and have it immutable
