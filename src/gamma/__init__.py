#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Global definitions of basic types and functions for use across alpha libraries
"""

import logging
from functools import wraps
from typing import *

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# noinspection PyShadowingBuiltins
_T = TypeVar("_T")
ListLike = Union[np.ndarray, pd.Series, Sequence[_T]]
MatrixLike = Union[
    np.ndarray, pd.Series, pd.DataFrame, Sequence[Union[_T, Sequence[_T]]]
]


def deprecated(message: str):
    """
    Decorator to mark functions as deprecated.

    It will result in a warning being logged when the function is used.

    :return: decorator; the decorated functions logs a warning message saying it is
      deprecated
    """

    def _deprecated_inner(func: callable) -> callable:
        @wraps(func)
        def new_func(*args, **kwargs) -> Any:
            """
            Function wrapper
            """
            message_header = f"Call to deprecated function {func.__name__}"
            if message is None:
                log.warning(message_header)
            else:
                log.warning(f"{message_header}: {message}")
            return func(*args, **kwargs)

        return new_func

    return _deprecated_inner
