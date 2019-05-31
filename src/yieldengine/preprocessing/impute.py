import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from yieldengine.transform import ColumnPreservingTransformer

log = logging.getLogger(__name__)


class SimpleImputerDF(ColumnPreservingTransformer[SimpleImputer]):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> SimpleImputer:
        return SimpleImputer(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        stats = self.base_transformer.statistics_
        if issubclass(stats.dtype.type, float):
            nan_mask = np.isnan(stats)
        else:
            nan_mask = [
                x is None or (isinstance(x, float) and np.isnan(x)) for x in stats
            ]
        return self.columns_in.delete(np.argwhere(nan_mask))
