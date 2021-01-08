"""
Learner inspection with standard local SHAP and global FACET metrics describing how
features combine to contribute to all model predictions.

The :class:`.LearnerInspector` class computes the shap values and the associated metrics
of a learner pipeline which has been fitted using cross-validation.
"""
from ._explainer import *
from ._inspection import *
