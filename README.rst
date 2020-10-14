.. image:: _static/facet_banner.png

|

`facet` is an open source library for human-explainable AI. It combines sophisticated
model inspection and model-based simulation to enable better explanations of your
machine learning models.

|azure_pypi| |azure_conda| |azure_devops_master_ci| |code_cov|
|python_versions| |code_style| |documentation_status|
|made_with_sphinx_doc| |License_badge|

Installation
---------------------

Facet supports both PyPI and Anaconda

Anaconda
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: RST

    conda install gamma-facet

Pip
~~~~~~~~~~~

.. code-block:: RST

    pip install gamma-facet

Quickstart
----------------------

Facet is composed of the following key components:

- **sklearndf**:
    An augmented version of scikit-learn with enhanced support for pandas dataframes
    and pipelining.

- **Enhanced machine learning workflow**:
    Facet delivers a robust and fail-safe pipelining
    workflow which allows you to easily impute and select your features as well as
    ranking a grid of different models "competing" against each other

- **Model Inspection**:
    Local explanations of features and their interactions make up a key
    component of understanding feature importance as well as feature interactions.
    This is based on a novel method which decomposes
    `SHAP values <https://shap.readthedocs.io/en/latest/>`_ into
    three vectors representing **synergy**, **redundancy**, and **independence**.

- **Model Simulation**:
    Use your trained model and the insights from the model inspection
    to conduct a historical simulation of any feature on your target in order to
    identify local optima.


Pipelining and Model Ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    # standard imports
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.model_selection import RepeatedKFold

    # some helpful imports from sklearndf
    from sklearndf.pipeline import RegressorPipelineDF
    from sklearndf.regression import RandomForestRegressorDF

    # relevant FACET imports
    from facet.data import Sample
    from facet.selection import LearnerRanker, LearnerGrid

    # load Boston housing dataset
    boston = load_boston()
    df = pd.DataFrame(data=boston.data, columns=boston.feature_names).assign(
        MEDIAN_HOUSE_PRICE=boston.target
    )

    # create FCAET sample object
    boston_obs = Sample(observations=df, target="MEDIAN_HOUSE_PRICE")

    # create pipeline for random forest regressor
    rforest_reg = RegressorPipelineDF(regressor=RandomForestRegressorDF(random_state=42))

    # define grid of models which are "competing" against each other
    rforest_grid = [
        LearnerGrid(
            pipeline=rforest_reg, learner_parameters={"min_samples_leaf": [8, 11, 15]}
        )
    ]

    # create repeated k-fold CV iterator
    rkf_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    # rank your models by performance (default is variance explained)
    ranker = LearnerRanker(grids=rforest_grid, cv=rkf_cv, n_jobs=-3).fit(sample=boston_obs)

    # get summary report
    print(ranker.summary_report())

.. code-block:: RST

    Rank  1: RandomForestRegressorDF, ranking_score=    0.722, scores_mean=    0.813,
    scores_std=   0.0455, parameters={regressor__min_samples_leaf=8}

    Rank  2: RandomForestRegressorDF, ranking_score=    0.707, scores_mean=    0.802,
    scores_std=   0.0471, parameters={regressor__min_samples_leaf=11}

    Rank  3: RandomForestRegressorDF, ranking_score=    0.693, scores_mean=    0.789,
    scores_std=   0.0481, parameters={regressor__min_samples_leaf=15}

Easy model inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Facet implements several model inspection methods for
`scikit-learn <https://scikit-learn.org/stable/index.html>`_ base learners.
Fundamentally, facet enables post-hoc model inspection by breaking down the interaction
effects of the variables that your model used for training:

- **Redundancy**
  identifies groups of variables that fully or partially duplicate each
  other and do not deliver any additional information to the machine learning model.
- **Synergy**
  provides visibility about how features contain complementary information
  with respect to the target and team up to predict outcomes by combining their
  information.

.. code-block:: Python

    # fit the model inspector
    from facet.inspection import LearnerInspector
    inspector = LearnerInspector()
    inspector.fit(crossfit=ranker.best_model_crossfit)

    # visualise redundancy as a matrix
    from pytools.viz.matrix import MatrixDrawer
    redundancy_matrix = inspector.feature_redundancy_matrix()
    MatrixDrawer(style="matplot%").draw(redundancy_matrix, title="Redundancy Matrix")

.. image:: _static/redundancy_matrix.png
    :width: 400

We can also better visualize redundancy as a dendrogram so we can identify clusters of features with redundancy.

.. code-block:: Python

    # visualise redundancy using a dendrogram
    from pytools.viz.dendrogram import DendrogramDrawer
    redundancy = inspector.feature_redundancy_linkage()
    DendrogramDrawer().draw(data=redundancy, title="Redundancy Dendrogram")

.. image:: _static/redundancy_dendrogram.png
    :width: 400

For feature synergy, we can get a similar picture

.. code-block:: Python

    # visualise synergy as a matrix
    synergy_matrix = inspector.feature_synergy_matrix()
    MatrixDrawer(style="matplot%").draw(synergy_matrix, title="Synergy Matrix")

.. image:: _static/synergy_matrix.png
    :width: 400

Please see the API documentation for more detail.


Simulation
~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    # FACET imports
    from facet.validation import BootstrapCV
    from facet.crossfit import LearnerCrossfit
    from facet.simulation import UnivariateUpliftSimulator
    from facet.simulation.partition import ContinuousRangePartitioner
    from facet.simulation.viz import SimulationDrawer

    # create bootstrap CV iterator
    bscv = BootstrapCV(n_splits=1000, random_state=42)

    # create a bootstrap CV crossfit for simulation using best model
    boot_crossfit = LearnerCrossfit(
        pipeline=ranker.best_model.native_estimator,
        cv=bscv,
        n_jobs=-3,
        verbose=False,
    ).fit(sample=boston_obs)

    SIM_FEAT = "LSTAT"
    simulator = UnivariateUpliftSimulator(crossfit=ranker.best_model_crossfit, n_jobs=3)

    # split the simulation range into equal sized partitions
    partitioner = ContinuousRangePartitioner()

    # run the simulation
    simulation = simulator.simulate_feature(name=SIM_FEAT, partitioner=partitioner)

    # visualise results
    SimulationDrawer().draw(data=simulation, title=SIM_FEAT)

.. image:: _static/simulation_output.png

.. raw:: html

    <p>Download the getting started tutorial and explore FACET for yourself by clicking
    here:
    <a href="https://mybinder.org" target="_blank">
    <img src="https://mybinder.org/badge_logo.svg"></a>
    </p>


Development Guidelines
---------------------------

TBD - link to long section in documentation.

Acknowledgements
---------------------------

GAMMA Facet is built on top of two popular packages for Machine
Learning:

The `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ learners and
pipelining make up implementation of the underlying algorithms. Moreover, we tried
to design the `facet` API to align with the scikit-learn API.

The `shap <https://github.com/slundberg/shap>`_ implementation is used to estimate the
shapley vectors which are being decomposed into the synergy, redundancy, and
independence vectors.

.. |azure_conda| image::
    :target:
.. |azure_pypi| image::
    :target:
.. |azure_devops_master_ci| image::
    :target:
.. |code_cov| image::
    :target:
.. |documentation_status| image::
    :target:

.. |python_versions| image:: https://img.shields.io/badge/python-3.7|3.8-blue.svg
    :target: https://www.python.org/downloads/release/python-380/

.. |code_style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
.. |made_with_sphinx_doc| image:: https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg
    :target: https://www.sphinx-doc.org/
.. |license_badge| image:: https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg
    :target: https://opensource.org/licenses/Apache-2.0
