"""
Machine learning workflows for advanced model selection and inspection.


Implements the following subpackages:

- :mod:`gamma.ml.selection`: model selection and hyperparameter tuning
- :mod:`gamma.ml.validation`: scikit-learn-style cross-validators for different \
    bootstrapping approaches
- :mod:`gamma.ml.crossfit`: a `crossfit` object manages multiple fits \
    of the same learner across all splits of a cross-validator, enabling a range of \
    approaches for model selection, inspection, and simulation/optimization (see \
    :mod:`gamma.yieldengine`)
- :mod:`gamma.ml.inspection`: explaining the interactions of a model's features \
    with each other, and with the target variable, based on the SHAP approach
"""

# todo: explain the basic workflow in the docstring:
#       LearnerRanker --> LearnerCrossfit --> LearnerInspector


from ._ml import Sample

__version__ = "1.3.0rc1"
