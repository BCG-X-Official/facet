"""
Machine learning library for advanced model selection, validation, and inspection.

Implements the following subpackages:

- :mod:`gamma.ml.selection`: simultaneous hyperparameter optimization for one or \
    more scikit-learn learners
- :mod:`gamma.ml.validation`: cross-validators for bootstrapping and stationary \
    bootstrapping (in the case of time series) - both not offered natively by \
    scikit-learn
- :mod:`gamma.ml.crossfit`: a unified approach to manage multiple fits \
    of the same learner across all splits of a cross-validator, enabling a range of \
    methods for model selection, inspection, and simiulation/optimization (see \
    :mod:`gamma.yieldengine`)
- :mod:`gamma.ml.inspection`: explaining the interactions of a model's features \
    with each other, and with the target variable, based on the SHAP approach
"""

import gamma.common.licensing as _licensing
from ._ml import Sample

_licensing.check_license(__package__)
