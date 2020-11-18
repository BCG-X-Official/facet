.. _contribution-guide:

Development Guidelines
======================================

Setup
-----------------------

Python environment
~~~~~~~~~~~~~~~~~~~~~~
There is a ``environment.yml`` provided in the repository root, which installs all required development
dependencies in the ``facet-development`` environment.

.. code-block:: RST

	conda env create -f environment.yml
	conda activate facet-develop

Pytest
~~~~~~~~~~~~~~~
Run ``pytest tests/`` from the facet root folder or use the PyCharm test runner. To measure coverage, use ``pytest --cov=src/facet tests/``. Note that the code coverage reports are also generated in the Azure Pipelines (see CI/CD section).

Note that you will need to set the PYTHONPATH to the ``src/`` directory by running ``export PYTHONPATH=./src/`` from the repository root.


Sphinx Documentation
~~~~~~~~~~~~~~~~~~~~~~~

The generated Sphinx documentation for facet is located at ``sphinx/build/html``. To build the documentation, ensure you have the following dependencies installed (these are included in the environment.yml):

- The Python packages sphinx=2.0.1 installed

- pydata-sphinx-theme=0.3.1 installed

- nbsphinx installed with ``conda install -c conda-forge nbsphinx=0.4.2``

- sphinx-autodoc-typehints installed with ``conda install -c conda-forge sphinx-autodoc-typehints=1.6``

Before building ensure you have Activated the ``facet-develop`` Conda environment. The following extensions are used in the conf.py Sphinx configuration script:

- `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ (external links to other documentations built with Sphinx: scikit-learn, numpy...)

- `viewcode <https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html>`_ to include source code in the documentation, and links to the source code from the objects documentation

- `imgmath <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_ to render math expressions in doc strings. Note that a local latex installation is required (e.g., `MiKTeX <https://miktex.org/>`_ for Windows)

To update the Sphinx documentation, run ``make html`` from within ``/sphinx``. By default this will only compile files that have been modified since the last compilation. To force compilation of the full documentation run ``make clean`` first.

**TODO**: To publish the documentation - finish this section


Git Guidelines
--------------------

For commits to GitHub, phrase commit comments as the completion of the sentence "This
commit will <...>", e.g.

.. code-block:: RST

	add method foo to class Bar

but not

.. code-block:: RST

	added method foo to class Bar


Documentation Guidelines
---------------------------


General guidelines
~~~~~~~~~~~~~~~~~~~~~~~

- The documentation is generated from docstrings in the source code

- Before writing your own documentation, take some time to study the documentation of the existing code and emulate the same style

- Describe not only what the code does, but also why, including the rationale for any design choices that may not be obvious

- Provide examples wherever this helps explain usage patterns


Docstring guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~
- A docstring is mandatory for all of the following entities in the source code, except when they are protected/private (i.e. the name starts with a leading _ character):

    - modules

    - classes

    - functions/methods

    - properties

    - attributes

- Docstrings are not necessary for non-public methods, but you should have a comment that describes what the method does (PEP 8)

- Docstrings must follow the reStructuredText syntax, the default syntax for Sphinx

- Write docstrings for functions and methods in the imperative style, e.g.,

.. code-block:: RST

	def fit():
	"""Fit the model."""

but not

.. code-block:: RST

	def fit():
	"""This is a function that fits the model."""

(too wordy and not imperative)


- Write docstrings for modules, classes, modules, and attributes starting with a descriptive phrase (as you would expect in a dictionary entry). Be concise and avoid unnecessary or redundant phrases. For example:

.. code-block:: RST

	class Inspector:
	    """
	    Explains the inner workings of a predictive model using the SHAP approach.

	    The inspector offers the following analyses:
	    - ...
	    - ...

but not

.. code-block:: RST

	class Inspector:
	    """
	    This is a class that provides the functionality to inspect models
	    ...

(too verbose, and explains the class in terms of its name which does not add any
information)

- Properties should be documented as if they were attributes, not as methods, e.g.,

.. code-block:: RST

	@property
	    def children(self) -> Foo:
	        """the child nodes of the tree"""
	        pass

but not

.. code-block:: RST

	@property
	    def foo(self) -> Foo:
	        """:return: the foo object"""
	        pass

- Start full sentences and phrases with a capitalised word and end each sentence with punctuation, e.g.,

``"""Fit the model"""``

but not

``"""fit the model"""``


- For multi-line docstrings, insert a line break after the leading triple quote and before the trailing triple quote, e.g.,

.. code-block:: RST

	def fit():
	    """
	    Fit the model.

	    Use the underlying estimator's ``fit`` method
	    to fit the model using the given training sample.

	    :param sample: training sample
	    """

but not

.. code-block:: RST

	def fit():
	    """Fit the model.

	    Use the underlying estimator's ``fit`` method
	    to fit the model using the given training sample.

	    :param sample: training sample"""

- For method arguments, return value, and class parameters, one must hint the type using the typing module. Do not specify the parameter types in the docstrings, e.g.,

.. code-block:: RST

	def f(x: int) -> float:
	   """
	   Do something.

	   :param x: input value
	   :return: output value

but not

.. code-block:: RST

	def f(x: int) -> float:
	   """
	   Do something.

	   :param int x: input value
	   :return float: output value


Converting notebooks to documentation with nbsphinx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- To hide a notebook cell from the generated documentation, add "nbsphinx": "hidden" to the metadata of the cell. To change the metadata of a cell, in the main menu of the jupyter notebook server, click on *View -> CellToolbar -> edit Metadata*, then click on edit Metadata in the top right part of the cell.
- To interpret a notebook cell as reStructuredText by nbsphinx, make a Raw NBConvert cell, then click on the jupyter notebook main menu to *View -> CellToolbar -> Raw Cell Format*, then choose ReST in the dropdown in the top right part of the cell.


Building and releasing facet
--------------------------------

Release & Version management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Facet version numbers follow the `Semantic versioning <https://semver.org/>`_ approach,
with the pattern ``MAJOR.MINOR.PATCH``. We are using
`punch <https://punch.readthedocs.io/en/latest/>`_ to increase the version numbers
for future releases.

To make a new deployment, you should:

1. Increase the version number with ``punch``:

	a. Ensure you have once fetched the ``release`` branch
	b. From ``develop`` git merge into ``release``
	c. From ``release``, run ``punch -p [major|minor|patch]`` to increase the version part of your choice
	d. Note that this will update the version number in ``setup.py`` and relevant parts of the documentation as well as commit this to the ``release`` branch
	e. Merge ``release`` back into ``develop`` and push both branches to deploy the update

2. PR from release to Master

	a. Open a PR from release to master to finalize the release - the Azure Pipelines must have passed for the release branch.


Conda Packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build
""""""""""""

Useful references:

- `Conda build tutorial <https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/building-conda-packages.html>`_
- `Conda build metadata reference <https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html>`_

Facet uses a combination of ``conda-build`` and ``make`` (both further explained below),
for which the necessary Conda build recipes are maintained under
``conda-build/meta.yaml``.

Build output will be stored in the ``dist/conda/`` directory (gitignored).

**Conda build recipes**

In this section, the structure of the conda-build recipe stored within ``conda-build/``
is explained.

The ``package`` section indicates the name of the resulting Conda package and its version.

.. code-block:: RST

	package:
		name: facet
		version: 1.0.0

When setting the version for a build, ``punch`` will update the version here - all other
conda-build specifications will refer to it dynamically by the ``PKG_VERSION`` variable.

The **source** section specifies from where the conda-build will acquire the sources
to build.

.. code-block:: RST

	source:
		git_url: https://github.com/bcg-gamma/facet/
		git_rev: refs/tags/{{PKG_VERSION}}

Note that using the ``PKG_VERSION`` here will always use the latest published version tag.

The **build** section indicates how the previously acquired code should be built:

.. code-block:: RST

	build:
		noarch: python
		script: "python -m pip install . --no-deps --ignore-installed -vv"

Note that setting the ``noarch: Python`` flag produces a pure Python, cross-platform
build. The command given to ``script`` indicates what ``conda-build`` will do to build the
underlying package: in this case it will pip install it using the ``setup.py`` in
the root of the repository. The ``--no-deps`` switch is passed, so that all
dependencies to other libraries are managed by Conda and not pip.


The **requirements** section specifies those dependencies that ``facet`` has:

.. code-block:: RST

	requirements:
		host:
			- pip
			- python={{ environ.get('FACET_V_PYTHON_BUILD', '3.7') }}
		run:
			- python>=3.6,<3.8
			- pandas{{ environ.get('FACET_V_PANDAS', '>=0.24') }}
			- numpy{{ environ.get('FACET_V_NUMPY', '>=1.16') }}
			- matplotlib{{ environ.get('FACET_V_MATPLOT', '>=3') }}
			- shap{{ environ.get('FACET_V_SHAP', '>=0.34') }}
			- scikit-learn{{ environ.get('FACET_V_SKLEARN', '>=0.21,<=0.22') }}
			- gamma-pytools=1.0
			- gamma-sklearndf=1.0
			- pyyaml>=5

The ``host`` section defines solely what is needed to carry out the build: Python and
pip.

The ``run`` section defines which Conda packages are required by ``facet`` at runtime.
You can see that we defined
environment variables such as ``V_FACET_PYTHON_BUILD``. This allows us to test a matrix
strategy of different combinations dependencies in our ``azure-pipelines.yml`` on
Azure DevOps. If the environment variable is not specified, the default value is given
in this section of the ``meta.yaml``. This setup helps us to detect version conflicts.

The **test** section specifies which tests should be carried out to verify a successful
build of the package:

.. code-block:: RST

    imports:
    - facet
        - facet.crossfit
        - facet.inspection
        - facet.selection
        - facet.validation
        - facet.simulation
    requires:
        - pytest=5.2
    commands:
        - python -c 'import facet;
          import os;
          assert facet.__version__ == os.environ["PKG_VERSION"]'

In this case, we want to check that all required packages can be imported successfully
and that the version of facet is aligned with the ``PKG_VERSION``.

**Makefile**

A common ``Makefile`` helps to orchestrate the facet build at a higher level, fully
relying on the Conda build recipes introduced above.

**Local Building on macOS**

As introduced above, local building of facet is done using the Makefile that will in
turn orchestrate ``conda-build``.

Please make sure to activate the ``facet-develop`` environment such that
``conda-build`` is available. When you are in the root of the ``facet`` directory,
you can build the package locally using

.. code-block:: RST

    make package

and delete the package using

.. code-block:: RST

    make clean

If successful, the ``dist/conda`` folder should contain the built Conda packages.

Publishing
"""""""""""""""

**TODO** - once published.


PyPI packages
~~~~~~~~~~~~~~~

Build
"""""""
As mentioned the previous section, the ``conda-build`` is using ``pip`` in order to
build the Conda package. This is using the standard ``setup.py`` required by PyPI. You
can read more about it
`here <https://packaging.python.org/tutorials/packaging-projects/>`_.

In order to locally install the package for testing, you can run:

.. code-block:: RST

    pip install -e .


Publishing
"""""""""""""""""

**TODO** - once published.




CI/CD
------------------

This project uses `Azure Devops <https://dev.azure.com/>`_ for CI/CD pipelines.
The pipelines are defined in the ``azure-pipelines.yml`` file and are divided into
two main stages.

Stage 1 - Development environment build and testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "Environment build & Pytest" stage performs the following steps:

- Checks out the ``facet`` repository at the develop branch
- Creates the ``facet-develop`` environment from the ``environment.yml``
- Installs the ``sklearndf`` and ``pytools`` dependencies
- Runs ``pytest`` and generates the code coverage reports for Azure DevOps. Note that these can be viewed on the Pipeline summary page.


Stage 2 - Matrix Strategy for Conda package build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "Test multiple conda environment builds" stage performs the following steps:

- Checks out the ``facet`` repository at the development branch
- Sets the environment variables of the ubuntu-vm as specified in the matrix strategy
- Runs ``make package`` for ``facet`` for each combination of the following matrix:

.. code-block:: RST

    strategy:
        matrix:
          Minimum dependencies:
            FACET_V_PYTHON_BUILD: '3.6'
            FACET_V_PANDAS: '==0.24'
            FACET_V_SKLEARN: '==0.21.*'
            FACET_V_JOBLIB: '==0.13'
            FACET_V_NUMPY: '==1.16'
            FACET_V_SHAP: '==0.34'
          Maximum dependencies:
            FACET_V_PYTHON_BUILD: '3.8'
            FACET_V_SKLEARN: '==0.23'
            FACET_V_PANDAS: '==1.0.0'
            FACET_V_NUMPY: '=>1.16'
            FACET_V_SHAP: '==0.35'
          Unconstrained dependencies:
            FACET_V_PYTHON_BUILD: '>=3.6'
            FACET_V_PANDAS: '=>0.24'
            FACET_V_SKLEARN: '=>0.21'
            FACET_V_JOBLIB: '=>0.13'
            FACET_V_NUMPY: '=>1.16'
            FACET_V_SHAP: '=>0.34'

Note that the environment variables set here are referenced in the
``conda-build/meta.yaml``. Testing this variety of package dependencies helps
to identify potential version conflicts.