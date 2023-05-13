"""
Type aliases for common use in the inspection package
"""

from typing import Callable, Union

import numpy as np
import pandas as pd

ModelFunction = Callable[
    [Union[pd.Series, pd.DataFrame, np.ndarray]], Union[float, pd.Series, np.ndarray]
]
