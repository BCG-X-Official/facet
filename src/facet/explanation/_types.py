"""
Type aliases and constants for common use in the ``facet.explanation`` package
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import catboost
else:
    try:
        import catboost
    except ImportError:
        from types import ModuleType

        catboost = ModuleType("catboost")
        catboost.Pool = type("Pool", (), {})

ArraysAny: TypeAlias = Union[npt.NDArray[Any], List[npt.NDArray[Any]]]
ArraysFloat: TypeAlias = Union[npt.NDArray[np.float64], List[npt.NDArray[np.float64]]]
CatboostPool: TypeAlias = catboost.Pool
XType: TypeAlias = Union[npt.NDArray[Any], pd.DataFrame, catboost.Pool]
YType: TypeAlias = Union[npt.NDArray[Any], pd.Series, None]
