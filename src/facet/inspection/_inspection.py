"""
Core implementation of :mod:`facet.inspection`
"""
import logging
from typing import List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeAlias

from pytools.api import AllTracker

from ..data import Sample

log = logging.getLogger(__name__)

__all__ = [
    "ShapPlotData",
]


#
# Type aliases
#

FloatArray: TypeAlias = npt.NDArray[np.float_]


#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


class ShapPlotData:
    """
    Data for use in SHAP plots provided by the
    `shap <https://shap.readthedocs.io/en/stable/>`__ package.
    """

    def __init__(
        self, shap_values: Union[FloatArray, List[FloatArray]], sample: Sample
    ) -> None:
        """
        :param shap_values: the shap values for all observations and outputs
        :param sample: (sub)sample of all observations for which SHAP values are
            available; aligned with param ``shap_values``
        """
        self._shap_values = shap_values
        self._sample = sample

    @property
    def shap_values(self) -> Union[FloatArray, List[FloatArray]]:
        """
        Matrix of SHAP values (number of observations by number of features)
        or list of shap value matrices for multi-output models.
        """
        return self._shap_values

    @property
    def features(self) -> pd.DataFrame:
        """
        Matrix of feature values (number of observations by number of features).
        """
        return self._sample.features

    @property
    def target(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Series of target values (number of observations)
        or matrix of target values for multi-output models
        (number of observations by number of outputs).
        """
        return self._sample.target


__tracker.validate()
