FACET is an open source library for human-explainable AI.
It combines sophisticated model inspection and model-based simulation to enable better
explanations of your supervised machine learning models.

FACET is composed of the following key components:


**Model Inspection**

FACET introduces a new algorithm to quantify dependencies and
interactions between features in ML models.
This new tool for human-explainable AI adds a new, global
perspective to the observation-level explanations provided by the
popular `SHAP <https://shap.readthedocs.io/en/stable/>`__ approach.
To learn more about FACET’s model inspection capabilities, see the
getting started example below.


**Model Simulation**

FACET’s model simulation algorithms use ML models for
*virtual experiments* to help identify scenarios that optimise
predicted outcomes.
To quantify the uncertainty in simulations, FACET utilises a range
of bootstrapping algorithms including stationary and stratified
bootstraps.
For an example of FACET’s bootstrap simulations, see the
quickstart example below.


**Enhanced Machine Learning Workflow**

FACET offers an efficient and transparent machine learning
workflow, enhancing
`scikit-learn <https://scikit-learn.org/stable/index.html>`__'s
tried and tested pipelining paradigm with new capabilities for model
selection, inspection, and simulation.
FACET also introduces
`sklearndf <https://github.com/BCG-Gamma/sklearndf>`__, an augmented
version of *scikit-learn* with enhanced support for *pandas* data
frames that ensures end-to-end traceability of features.


.. Begin-Badges

|pypi| |conda| |python_versions| |code_style| |made_with_sphinx_doc| |License_badge|

.. End-Badges

License
---------------------------

FACET is licensed under Apache 2.0 as described in the
`LICENSE <https://github.com/BCG-Gamma/facet/blob/develop/LICENSE>`_ file.

.. Begin-Badges

.. |conda| image:: https://anaconda.org/bcg_gamma/gamma-facet/badges/version.svg
    :target: https://anaconda.org/BCG_Gamma/gamma-facet

.. |pypi| image:: https://badge.fury.io/py/gamma-facet.svg
    :target: https://pypi.org/project/gamma-facet/

.. |python_versions| image:: https://img.shields.io/badge/python-3.6|3.7|3.8-blue.svg
   :target: https://www.python.org/downloads/release/python-380/

.. |code_style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |made_with_sphinx_doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
   :target: https://bcg-gamma.github.io/facet/index.html

.. |license_badge| image:: https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg
   :target: https://opensource.org/licenses/Apache-2.0

.. End-Badges