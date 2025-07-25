"""
Factory classes for common SHAP explainers.

Provides a unified API for creating SHAP explainers provided by the :mod:`shap` package.

Used by :class:`.LearnerInspector` and :class:`.FunctionInspector` to create explainers
for the learner or function under inspection.
"""

from ._explanation import *
