"""
Meta-estimator containing a fitted estimator for each cross-validation training
split; used as the basis for learner selection and inspection.

:class:`.LearnerCrossfit` encapsulates a fully trained pipeline.
It contains a :class:`~.sklearndf.LearnerPipelineDF` (preprocessing and learner),
a dataset given by a :class:`.Sample` object, and a
cross-validator.
The pipeline is fitted accordingly.
"""

from ._crossfit import *
