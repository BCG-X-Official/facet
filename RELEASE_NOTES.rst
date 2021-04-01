Release Notes
=============

FACET 1.0
---------

1.0.3
~~~~~

- FIX: restrict package requirements to *gamma-pytools* 1.0.* and *sklearndf* 1.0.x, since FACET 1.0 is not compatible with *gamma-pytools* 1.1.* 

1.0.2
~~~~~

This is a maintenance release focusing on enhancements to the CI/CD pipeline and bug
fixes.

- API: add support for :mod:`shap` 0.36 and 0.37 via a new :class:`.BaseExplainer`
  stub class
- FIX: apply color scheme to the histogram section in :class:`.SimulationMatplotStyle`
- BUILD: add support for numpy 1.20
- BUILD: updates and changes to the CI/CD pipeline


1.0.1
~~~~~

Initial release.
