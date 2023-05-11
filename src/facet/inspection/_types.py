"""
Type aliases for common use in the inspection package
"""

from typing import Callable, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

ModelFunction = Callable[
    [Union[pd.DataFrame, npt.NDArray[np.float_]]],
    Union[pd.Series, npt.NDArray[np.float_], float],
]
