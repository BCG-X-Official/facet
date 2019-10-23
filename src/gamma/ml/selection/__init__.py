"""
ModelPipelineDF selection and hyperparameter optimisation.

:class:`ParameterGrid` encapsulates a :class:`gamma.ml.ModelPipelineDF` and a grid of
hyperparameters.

:class:`BaseLearnerRanker` selects the best pipeline and parametrisation based on the
pipeline and hyperparameter choices provided as a list of :class:`ModelGrid`.
"""
from ._core import *
