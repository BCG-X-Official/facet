import pandas as pd
from typing import List
import numpy as np


class Sample:
    def __init__(
        self, sample: pd.DataFrame, features: List[str] = None, target: str = None
    ) -> None:
        if sample is None or not (type(sample) == pd.DataFrame):
            raise ValueError("Expected 'sample' to be a pd.DataFrame")

        if features is None and target is None:
            raise ValueError("Either one of features, target need to be specified")

        if features is None:
            # in this case, target must be defined - use it to define features
            self.__target = target
            self.__features = [c for c in list(sample.columns) if c != target]
        else:
            self.__features = features

        if target is None:
            # in this case, features must be defined - use it to define target

            # get a list of all columns not in features...
            target_candidates = [c for c in list(sample.columns) if c not in features]

            # more than 1? we only expect to have one target variable - here it is unclear which it is
            if len(target_candidates) > 1:
                raise ValueError(
                    "'target' variable not defined, but given list 'features' "
                    "leaves more than one column out -> can't infer a single target column"
                )
            # only 1? set it
            self.__target = target_candidates[0]
        else:
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
