import pandas as pd
from boruta import BorutaPy

from yieldengine.feature.transform import (
    ColumnPreservingTransformer,
    NumpyOnlyTransformer,
)


class BorutaDF(NumpyOnlyTransformer[BorutaPy], ColumnPreservingTransformer[BorutaPy]):
    @classmethod
    def _make_base_transformer(cls, **kwargs) -> BorutaPy:
        return BorutaPy(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in[self.base_transformer.support_]
