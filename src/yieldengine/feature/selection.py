from typing import *

import pandas as pd
from boruta import BorutaPy

from yieldengine.feature.transform import NumpyOnlyTransformer


class BorutaDF(NumpyOnlyTransformer[BorutaPy]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> BorutaPy:
        return BorutaPy(**kwargs)

    @property
    def columns_out(self) -> pd.Index:
        return self.columns_in[self.base_transformer.support_]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> None:
        super().fit(X=X.copy(), y=None if y is None else y.copy(), **fit_params)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params
    ) -> pd.DataFrame:
        return super().fit_transform(
            X=X.copy(), y=None if y is None else y.copy(), **fit_params
        )
