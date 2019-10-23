"""
Cross-validation.

:class:`BootstrapCV` performs bootstrap sampling.

:class:`StationaryBootstrapCV` implements the static bootstrap for time series.
"""
from ._validation import *

_DEPRECATED = {"CircularCV"}

__all__ = [member for member in _validation.__all__ if member not in _DEPRECATED]
