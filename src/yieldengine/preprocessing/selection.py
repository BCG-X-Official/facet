import pandas as pd
from boruta import BorutaPy

from yieldengine.df.transform import ColumnPreservingTransformer, NumpyOnlyTransformer


class BorutaDF(NumpyOnlyTransformer[BorutaPy], ColumnPreservingTransformer[BorutaPy]):
    def __init__(
        self,
        estimator,
        n_estimators=1000,
        perc=100,
        alpha=0.05,
        two_step=True,
        max_iter=100,
        random_state=None,
        verbose=0,
        **kwargs
    ):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            two_step=two_step,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

    @classmethod
    def _make_base_transformer(cls, **kwargs) -> BorutaPy:
        return BorutaPy(**kwargs)

    def _get_columns_out(self) -> pd.Index:
        return self.columns_in[self.base_transformer.support_]
