"""
Machine learning workflows for advanced model selection and
inspection.


Implements the following subpackages:

- :mod:`facet.selection`: model selection and hyperparameter-tuning
- :mod:`facet.validation`: scikit-learn-style cross-validators \
for different bootstrapping approaches
- :mod:`facet.crossfit`: a `crossfit` object manages multiple
fits of the same learner across all splits of a cross-validator, \
enabling a range of \
approaches for model selection, inspection, and simulation/
optimization
- :mod:`facet.inspection`: explaining the interactions of a model's \
features with each other, and with the target variable, based on the SHAP \
approach
- :mod:`facet.simulation`: simulating variables using a predictive \
model
"""

# todo: explain the basic workflow in the docstring:
#       LearnerRanker --> LearnerCrossfit --> LearnerInspector


from ._facet import Sample

__version__ = "1.0.0"
