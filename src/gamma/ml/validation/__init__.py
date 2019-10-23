"""
Cross-validation.

:class:`BootstrapCV` performs bootstrap sampling.

:class:`CircularCV` class performs cross-validation with a fixed
test_ratio with a fix size window shifting at a constant pace = 1/num_splits.
"""
from ._core import *
