"""
Type aliases for common use in the ``facet`` package
"""

from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from typing_extensions import TypeAlias

# a function representing a model to be inspected
ModelFunction: TypeAlias = Callable[
    [Union[pd.Series, pd.DataFrame, npt.NDArray[np.float_]]],
    Union[pd.Series, npt.NDArray[np.float_], float],
]

# a supervised learner in scikit-learn
NativeSupervisedLearner: TypeAlias = Union[RegressorMixin, ClassifierMixin]
