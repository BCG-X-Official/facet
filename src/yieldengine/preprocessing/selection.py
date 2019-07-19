"""
Select relevant features.

Given a dataset and an estimator,
the Boruta method selects the features that perform better than noise for the problem.
See `BorutaPy
<https://github.com/scikit-learn-contrib/boruta_py>`_.

:class:`BorutaDF` wraps
:class:`BorutaPy <https://github.com/scikit-learn-contrib/boruta_py>` with dataframes
as input and output.
"""

import pandas as pd
from boruta import BorutaPy

from yieldengine.df.transform import ColumnPreservingTransformer, NDArrayTransformerDF

__all__ = ["BorutaDF"]


class BorutaDF(NDArrayTransformerDF[BorutaPy], ColumnPreservingTransformer[BorutaPy]):
    """
    Feature Selection with the Boruta method with dataframes as input and output.

    Wrap the boruta class :class:`BorutaPy` for feature selection and return a
    dataframe.

    ``self.fit`` accepts a dataframe and ``self.transform`` returns a dataframe.

    The parameters are the parameters from the boruta :class:`BorutaPy`. For
    convenience we list below the description of the parameters as they appear in
    https://github.com/scikit-learn-contrib/boruta_py.

    :param estimator: object
        A supervised learning estimator, with a 'fit' method that returns the
        `feature_importances_` attribute. Important features must correspond to
        high absolute values in the `feature_importances_`.
    :param n_estimators: int or string, default = 1000
        If int sets the number of estimators in the chosen ensemble method.
        If 'auto' this is determined automatically based on the size of the
        dataset. The other parameters of the used estimators need to be set
        with initialisation.
    :param perc: int, default = 100
        Instead of the max we use the percentile defined by the user, to pick
        our threshold for comparison between shadow and real features. The max
        tend to be too stringent. This provides a finer control over this. The
        lower perc is the more false positives will be picked as relevant but
        also the less relevant features will be left out. The usual trade-off.
        The default is essentially the vanilla Boruta corresponding to the max.
    :param alpha: float, default = 0.05
        Level at which the corrected p-values will get rejected in both
        correction steps.
    :param two_step: Boolean, default = True
        If you want to use the original implementation of Boruta with Bonferroni
        correction only set this to False.
    :param max_iter: int, default = 100
        The number of maximum iterations to perform.
    :param random_state: int, RandomState instance or None; default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    :param verbose: int, default=0
        Controls verbosity of output:
        - 0: no output
        - 1: displays iteration number
        - 2: which features have been selected already
    """

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
    ) -> None:
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
