.. image:: _static/Gamma_Facet_Logo_RGB_LB.svg
    :alt: GAMMA FACET
    :width: 400
    :class: padded-logo

Facet is an open source library for human-explainable AI. It combines sophisticated
model inspection and model-based simulation to enable better explanations of your
machine learning models.

TODO - add git badges as substitutions

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


Pipelining & Model Ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearndf.pipeline import RegressorPipelineDF
    from sklearndf.regression import RandomForestRegressorDF

    # Relevant facet imports
    from facet import Sample
    from facet.selection import LearnerRanker, LearnerGrid
    from facet.validation import BootstrapCV

    # Load boston housing dataset
    boston = load_boston()
    TARGET = 'MEDIAN_HOUSE_PRICE'
    df = pd.DataFrame(
            data=boston.data,
            columns=boston.feature_names).assign(MEDIAN_HOUSE_PRICE=boston.target)
    sample = Sample(observations=df, target=TARGET)


    rf_pipeline = RegressorPipelineDF(
        regressor=RandomForestRegressorDF(random_state=42)
    )

    # Define grid of models which are "competing" against each other
    grid = [
        LearnerGrid(
            pipeline=rf_pipeline,
            learner_parameters={
            "min_samples_leaf": [8, 11, 15]
    }
        )
    ]

    cv = BootstrapCV(n_splits=10, random_state=42)

    # Rank your models by performance
    ranker = LearnerRanker(
        grids=grid, cv=cv, n_jobs=-3
    ).fit(sample=sample)

    # Get your summary report
    ranker.summary_report()

.. code-block:: RST

    Rank  1: RandomForestRegressorDF, ranking_score=    0.673, scores_mean=    0.793,
         scores_std=   0.0598, parameters={regressor__min_samples_leaf=8}

    Rank  2: RandomForestRegressorDF, ranking_score=    0.666, scores_mean=     0.783,
             scores_std=   0.0589, parameters={regressor__min_samples_leaf=11}

    Rank  3: RandomForestRegressorDF, ranking_score=    0.659, scores_mean=    0.771,
             scores_std=   0.0561, parameters={regressor__min_samples_leaf=15}

Easy model inspection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Facet implements a number of model inspection methods for
`scikit-learn <https://scikit-learn.org/stable/index.html>`_ base learners.
Fundamentally, facet enables post-hoc model inspection by breaking down the interaction
effects of the variables that your model used for Training:

- **Redundancy**
  identifies groups of variables that fully or partially duplicate each
  other and do not deliver any additional information to the machine learning model.
- **Synergy**
  provides visibility about how features contain complementary information
  with respect to the target and team up to predict outcomes by combining their
  information.

.. code-block:: Python

    from facet.inspection import LearnerInspector
    from pytools.viz.matrix import MatrixDrawer

    inspector = LearnerInspector()
    inspector.fit(crossfit=ranker.best_model_crossfit)
    MatrixDrawer(style="matplot%").draw(inspector.feature_redundancy_matrix(),
                                        title="Redundancy Matrix")

.. image:: _static/redundancy_matrix.png
    :width: 400

We can also better visualize redundancy as a dendrogram so we can identify clusters of features with redundancy.

.. code-block:: Python

    from pytools.viz.dendrogram import DendrogramDrawer

    redundancy = inspector.feature_redundancy_linkage()
    DendrogramDrawer().draw(data=redundancy, title='Redundancy Dendrogram')

.. image:: _static/redundancy_dendrogram.png
    :width: 400

For feature synergy, we can get a similar picture

.. code-block:: Python

    synergy_matrix = inspector.feature_synergy_matrix()
    MatrixDrawer(style="matplot%").draw(synergy_matrix, title="Synergy Matrix")

.. image:: _static/synergy_matrix.png
    :width: 400

Please see the API documentation for more detail.


Simulation
~~~~~~~~~~~~~~~~~~

.. code-block:: Python

    from facet.simulation import UnivariateUpliftSimulator
    from facet.simulation.partition import ContinuousRangePartitioner
    from facet.simulation.viz import SimulationDrawer

    SIM_FEAT = "LSTAT"
    simulator = UnivariateUpliftSimulator(crossfit = ranker.best_model_crossfit, n_jobs=3)

    # Split the simulation range into equal sized partitions
    partitioner = ContinuousRangePartitioner()

    simulation = simulator.simulate_feature(name=SIM_FEAT, partitioner = partitioner)

    SimulationDrawer().draw(
        data=simulation, title=SIM_FEAT
    )

.. image:: _static/simulation_output.png

The notebook can be `downloaded here <../../notebooks/Boston_getting_started_example.ipynb>`_.


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

