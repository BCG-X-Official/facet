"""
Type aliases and constants for common use in the ``facet.explanation`` package
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    import catboost
else:
    try:
        import catboost
    except ImportError:
        from types import ModuleType

        catboost = ModuleType("catboost")
        catboost.Pool = type("Pool", (), {})

ArraysAny: TypeAlias = npt.NDArray[Any] | list[npt.NDArray[Any]]
ArraysFloat: TypeAlias = npt.NDArray[np.float64] | list[npt.NDArray[np.float64]]
CatboostPool: TypeAlias = catboost.Pool
XType: TypeAlias = npt.NDArray[Any] | pd.DataFrame | catboost.Pool
YType: TypeAlias = npt.NDArray[Any] | pd.Series | None
