"""
Type aliases for common use in the ``facet`` package
"""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin

# a function representing a model to be inspected
ModelFunction: TypeAlias = Callable[
    [pd.Series | pd.DataFrame | npt.NDArray[np.float64]],
    pd.Series | npt.NDArray[np.float64] | float,
]

# a supervised learner in scikit-learn
NativeSupervisedLearner: TypeAlias = RegressorMixin | ClassifierMixin
