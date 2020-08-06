"""
Model selection and hyperparameter optimisation.

:class:`.LearnerGrid` encapsulates a :class:`.LearnerPipelineDF` and a grid of
hyperparameters.

:class:`.LearnerRanker` selects the best pipeline and parametrisation based on the
pipeline and hyperparameter choices provided as a list of :class:`.LearnerGrid`.
"""
from ._selection import *
