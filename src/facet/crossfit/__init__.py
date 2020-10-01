"""
Meta-estimators that fit an estimator multiple times for all splits of a
cross-validator, as the basis for model evaluation and inspection.

:class:`LearnerCrossfit` encapsulates a fully trained pipeline.
It contains a :class:`.ModelPipelineDF` (preprocessing + estimator),
a dataset given by a
:class:`yieldengine.Sample` object and a
cross-validation calibration. The pipeline is fitted accordingly.
"""

from ._crossfit import *
