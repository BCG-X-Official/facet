"""
Shap calculations for functions
"""

import logging
from typing import Generic, List, Optional, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

from pytools.api import AllTracker, inheritdoc

from ..._types import ModelFunction
from ...explanation.base import ExplainerFactory
from ._shap import ShapCalculator

log = logging.getLogger(__name__)

__all__ = [
    "FunctionShapCalculator",
]


#
# Type variables
#

T_ModelFunction = TypeVar("T_ModelFunction", bound=ModelFunction)

#
# Ensure all symbols introduced below are included in __all__
#

__tracker = AllTracker(globals())


#
# Class definitions
#


@inheritdoc(match="""[see superclass]""")
class FunctionShapCalculator(ShapCalculator[T_ModelFunction], Generic[T_ModelFunction]):
    """
    Calculate SHAP values for a function.
    """

    # defined in superclass, repeated here for Sphinx:
    model: T_ModelFunction
    explainer_factory: ExplainerFactory[T_ModelFunction]
    shap_: Optional[pd.DataFrame]
    feature_index_: Optional[pd.Index]
    n_jobs: Optional[int]
    shared_memory: Optional[bool]
    pre_dispatch: Optional[Union[str, int]]
    verbose: Optional[int]

    @property
    def input_names(self) -> None:
        """Always ``None``, since functions require no fixed names for their inputs."""
        return None

    @property
    def output_names(self) -> List[str]:
        """[see superclass]"""
        try:
            return [self.model.__name__]
        except AttributeError:
            return ["output"]

    def _convert_shap_to_df(
        self,
        raw_shap_tensors: List[npt.NDArray[np.float_]],
        observation_idx: pd.Index,
        feature_idx: pd.Index,
    ) -> List[pd.DataFrame]:
        return self._convert_raw_shap_to_df(
            raw_shap_tensors=raw_shap_tensors,
            observation_idx=observation_idx,
            feature_idx=feature_idx,
        )
