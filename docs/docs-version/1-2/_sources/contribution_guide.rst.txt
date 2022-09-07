.. _contribution-guide:

Development Guidelines
======================================

Setup
-----------------------

Python environment
~~~~~~~~~~~~~~~~~~~~~~
There is an ``environment.yml`` provided in the repository root, which installs all
required development dependencies in the ``facet-develop`` environment.

.. code-block:: sh

    conda env create -f environment.yml
    conda activate facet-develop


Pre-commit hooks
~~~~~~~~~~~~~~~~~~~~
This project uses a number of pre-commit hooks such as black and flake8 to enforce
uniform coding standards in all commits. Before committing code, please run

.. code-block:: sh

    pre-commit install

You can use ``pre-commit run`` to manually run the pre-commit hooks from the command
line.


Pytest
~~~~~~~~~~~~~~~
Run ``pytest tests/`` from the facet root folder or use the PyCharm test runner. To
measure coverage, use ``pytest --cov=src/facet tests/``. Note that the code coverage
reports are also generated in the Azure Pipelines (see CI/CD section).

Note that you will need to set the PYTHONPATH to the ``src/`` directory by
running ``export PYTHONPATH=./src/`` from the repository root.




Git Guidelines
--------------------

For commits to GitHub, phrase commit comments as the completion of the sentence *This
commit will …*, e.g.

.. code-block:: RST

    add method foo to class Bar

but not

.. code-block:: RST

    added method foo to class Bar


Documentation
---------------------------

This section provides a general guide to the documentation of FACET, including
docstrings, Sphinx, the README and tutorial notebooks.

Docstrings
~~~~~~~~~~~

The API documentation is generated from docstrings in the source code. Before writing
your own, take some time to study the existing code documentation and emulate the same
style. Describe not only what the code does, but also why, including the rationale for
any design choices that may not be obvious. Provide examples wherever this helps
explain usage patterns.

- A docstring is mandatory for all of the following entities in the source code,
  except when they are protected/private (i.e. the name starts with a leading `_`
  character):

  - modules

  - classes

  - functions/methods

  - properties

  - attributes

- Docstrings are not necessary for non-public methods, but you should have a comment
  that describes what the method does.

- Docstrings must use *reStructuredText* syntax, the default syntax for Sphinx.

- Write docstrings for functions and methods in the imperative style, e.g.,

  .. code-block:: python

      def fit():
      """Fit the model."""

  but not

  .. code-block:: python

      def fit():
      """This is a function that fits the model."""

  which is too wordy and not imperative.


- Write docstrings for modules, classes, modules, and attributes starting with a 
  descriptive phrase (as you would expect in a dictionary entry). Be concise and avoid
  unnecessary or redundant phrases.
  For example:

  .. code-block:: python

      class Inspector:
          """
          Explains the inner workings of a predictive model using the SHAP approach.

          The inspector offers the following analyses:
          - ...
          - ...

  but not

  .. code-block:: python

      class Inspector:
          """
          This is a class that provides the functionality to inspect models
          ...

  as this is too verbose, and explains the class in terms of its name which does not add
  any information.

- Properties should be documented as if they were attributes, not as methods, e.g.,

  .. code-block:: python

      @property
      def children(self) -> Foo:
          """The child nodes of the tree"""
          pass

  but not

  .. code-block:: python

      @property
      def foo(self) -> Foo:
          """:return: the foo object"""
          pass

- Start full sentences and phrases with a capitalised word and end each sentence with 
  punctuation, e.g.,

  .. code-block:: python

    """Fit the model."""

  but not

  .. code-block:: python

    """fit the model"""


- For multi-line docstrings, insert a line break after the leading triple quote and before
  the trailing triple quote, e.g.,

  .. code-block:: python

    def fit():
        """
        Fit the model.

        Use the underlying estimator's ``fit`` method
        to fit the model using the given training sample.

        :param sample: training sample
        """

  but not

  .. code-block:: python

    def fit():
        """Fit the model.

        Use the underlying estimator's ``fit`` method
        to fit the model using the given training sample.

        :param sample: training sample"""

- For method arguments, return value, and class parameters, one must hint the type using 
  the typing module. Do not specify the parameter types in the docstrings, e.g.,

  .. code-block:: python

    def f(x: int) -> float:
       """
       Do something.

       :param x: input value
       :return: output value

  but not

  .. code-block:: python

    def f(x: int) -> float:
       """
       Do something.

       :param int x: input value
       :return float: output value


Sphinx Build
~~~~~~~~~~~~~~~~~~~~~~~

Documentation for FACET is built using `sphinx <https://www.sphinx-doc.org/en/master/>`_.
Before building the documentation ensure the ``facet-develop`` environment is active as
the documentation build has a number of key dependencies specified in the
``environment.yml`` file, specifically:

- ``sphinx``
- ``pydata-sphinx-theme``
- ``nbsphinx``
- ``sphinx-autodoc-typehints``

To generate the Sphinx documentation locally navigate to ``/sphinx`` and run

.. code-block:: sh

    python make.py html

By default this will clean any previous build. The generated Sphinx
documentation for FACET can then be found at ``sphinx/build/html``.

Documentation versioning is managed via the release process - see the section on
building and releasing FACET.

The ``sphinx`` folder in the root directory contains the following:

- a ``make.py`` script for executing the documentation build via python.

- a ``source`` directory containing predefined ``.rst`` files for the documentation
  build and other required elements, see below for more details.

- an ``auxiliary`` directory which contains the notebook used in the quickstart as well
  as a template notebook to be used when generating new tutorials to be added to the
  documentation. Note this is kept separate as it is used to generate the example for
  the repository `README.rst`, which is the included in the documentation build.


The ``sphinx/source`` folder contains:

- a ``conf.py`` script that is the `build configuration file 
  <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_ needed to customize the
  input and output behavior of the Sphinx documentation build (see below for further
  details).

- a ``tutorials`` directory that contains all the notebooks (and supporting data) used in
  the documentation build. Note that as some notebooks take a little while to generate, the
  notebooks are currently committed with cell output. This may change in the future where
  notebooks are run as part of the sphinx build.

- the base ``.rst`` files used for the documentation build, which are:

  *   ``index.rst``: definition of the high-level documentation structure which mainly
      references the other rst files in this directory.

  *   ``tutorials.rst``: a tutorial overview that incorporates the tutorial notebooks
      from the ``tutorials`` directory.

  *   ``contribution_guide.rst``: detailed information on building and releasing FACET.

  *   ``faqs.rst``: contains guidance on bug reports/feature requests, how to contribute
      and answers to frequently asked questions including small code snippets.

  *   ``about_us.rst``: description of the team behind open-sourcing FACET.

  *   ``api_landing.rst``: for placing any API landing page preamble for documentation
      as needed. This information will appear on the API landing page in the
      documentation build after the short description in ``src/__init__.py``. This file
      is included in the documentation build via the ``custom-module-template.rst``.

- ``_static`` contains additional material used in the documentation build (mainly
  figures) but also some formatting control:

  *   ``team_contributors``: contains photos for the FACET team.

  *   ``icons``: contains the icons used in describing the main elements of FACET
      in the documentation getting started page.

  *   ``css/facet.css``: contains additional customization for the display of HTML
      elements in the documentation build.

- ``_templates`` contains the ``autosummary.rst`` which relies on the 
  ``custom-module-template.rst`` and ``custom-class-template.rst`` from
  ``pytools/tree/develop/sphinx/source/_templates`` which is used in
  generating/formatting the modules and classes for the API documentation.

The two key scripts are ``make.py`` and ``conf.py``. The base configuration for these
scripts can be found in 
`pytools/sphinx <https://github.com/BCG-Gamma/pytools/tree/develop/sphinx>`_.
The reason for this was to minimise code given the standardization of the documentation
build across multiple packages.

**make.py**: All base configuration comes from ``pytools/sphinx/base/make_base.py`` and
this script includes defined commands for key steps in the documentation build. Briefly,
the key steps for the documentation build are:

- **Clean**: remove the existing documentation build.

- **FetchPkgVersions**: fetch the available package versions with documentation.

- **ApiDoc**: generate API documentation from sources.

- **Html**: run Sphinx build to generate HTMl documentation.

The two other commands are **Help** and **PrepareDocsDeployment**, the latter of which
is covered below under Building and releasing FACET.

**conf.py**: All base configuration comes from ``pytools/sphinx/base/conf_base.py``. This
`build configuration file <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_
is a requirement of Sphinx and is needed to customize the input and output behavior of
the documentation build. In particular, this file highlights key extensions needed in
the build process, of which some key ones are as follows:

- `intersphinx <https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html>`_ 
  (external links to other documentations built with Sphinx: scikit-learn, numpy...).

- `viewcode <https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html>`_ to 
  include source code in the documentation, and links to the source code from the objects 
  documentation.

- `imgmath <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`_ to render 
  math expressions in doc strings. Note that a local latex installation is required (e.g., 
  `MiKTeX <https://miktex.org/>`_ for Windows).


README
~~~~~~~

The README file for the repo is .rst format instead of the perhaps more traditional
markdown format. The reason for this is the ``README.rst`` is included as the quick start
guide in the documentation build. This helped minimize code duplication. However,
there are a few key points to be aware of:

- The README has links to figures, logos and icons located in the ``sphinx/source/_static`` 
  folder. To ensure these links are correct when the documentation is built, they are 
  altered and then the contents of the ``README.rst`` is incorporated into the 
  ``getting_started.rst`` which is generated during the build and can be found in 
  ``sphinx/source/getting_started``.

- The quick start guide based on the ``Boston_getting_started_example.ipynb`` notebook in 
  the ``sphinx/auxiliary`` folder is not automatically included (unlike all the other 
  tutorials). For this reason any updates to this example in the README need to be 
  reflected in the source notebook and vice-versa.


Tutorial Notebooks
~~~~~~~~~~~~~~~~~~~

Notebooks are used as the basis for detailed tutorials in the documentation. Tutorials
created for documentation need to be placed in ``sphinx/source/tutorial`` folder.

If you intend to create a notebook for inclusion in the documentation please note the
following:

- The notebook should conform to the standard format employed for all notebooks included in 
  the documentation. This template (``Facet_sphinx_tutorial_template.ipynb``) can be found 
  in ``sphinx/auxiliary``.

- When creating/revising a tutorial notebook with the development environment the following 
  code should be added to a cell at the start of the notebook. This will ensure your local 
  clones (and any changes) are used when running the notebook. The jupyter notebook should  
  also be instigated from within the ``facet-develop`` environment.

  .. code-block:: python

      def _set_paths() -> None:

          # set the correct path when launched from within PyCharm

          module_paths = ["pytools", "facet", "sklearndf"]

          import sys
          import os

          if "cwd" not in globals():
              # noinspection PyGlobalUndefined
              global cwd
              cwd = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
              os.chdir(cwd)
          print(f"working dir is '{os.getcwd()}'")

          for module_path in module_paths:
              if module_path not in sys.path:
                  sys.path.insert(0, os.path.abspath(f"{cwd}/{os.pardir}/{module_path}/src"))
              print(f"added `{sys.path[0]}` to python paths")

      _set_paths()

      del _set_paths



- If you have a notebook cell you wish to be excluded from the generated documentation, add 
  ``"nbsphinx": "hidden"`` to the metadata of the cell. To change the metadata of a cell, 
  in the main menu of the jupyter notebook server, click on *View -> CellToolbar -> edit 
  metadata*, then click on edit Metadata in the top right part of the cell. The modified 
  metadata would then look something like:

  .. code-block:: json

      {
        "nbsphinx": "hidden"
      }

- To interpret a notebook cell as reStructuredText by nbsphinx, make a Raw NBConvert cell, 
  then click on the jupyter notebook main menu to *View -> CellToolbar -> Raw Cell Format*, 
  then choose ReST in the dropdown in the top right part of the cell.

- The notebook should be referenced in the ``tutorials.rst`` file with a section structure 
  as follows:

  .. code-block:: RST

      NAME OF NEW TUTORIAL
      *****************************************************************************

      Provide a brief description of the notebook context, such as; regression or
      classification, application (e.g., disease prediction), etc.

      - Use bullet points to indicate what key things the reader will learn (i.e., key takeaways).

      Add a short comment here and direct the reader to download the notebook:
      :download:`here <tutorial/name_of_new_tutorial_nb.ipynb>`.

      .. toctree::
          :maxdepth: 1

          tutorial/name_of_new_tutorial_nb

- The source data used for the notebook should also be added to the tutorial folder unless 
  the file is extremely large and/or can be accessed reliably another way.

- For notebooks involving simulation studies, or very long run times consider saving 
  intermediary outputs to make the notebook more user-friendly. Code the produces the 
  output should be included as a markdown cell with code designated as python to ensure 
  appropriate formatting, while preventing the cell from executing should the user run
  all cells.


Package builds
--------------------------------

The build process for the PyPI and conda distributions uses the following key
files:

- ``make.py``: generic Python script for package builds. Most configuration is imported
  from pytools `make.py <https://github.com/BCG-Gamma/pytools/blob/develop/make.py>`__
  which is a build script that wraps the package build, as well as exposing the matrix
  dependency definitions specified in the ``pyproject.toml`` as environment variables.
- ``pyproject.toml``: metadata for PyPI, build settings and package dependencies.
- ``tox.ini``: contains configurations for tox, testenv, flake8, isort, coverage report, 
  and pytest.
- ``condabuild/meta.yml``: metadata for conda, build settings and package dependencies.

Versioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FACET version numbering follows the `semantic versioning <https://semver.org/>`_
approach, with the pattern ``MAJOR.MINOR.PATCH``.
The version can be bumped in the ``src/__init__.py`` by updating the
``__version__`` string accordingly.

PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyPI project metadata, build settings and package dependencies
are obtained from ``pyproject.toml``. To build and then publish the package to PyPI,
use the following commands:

.. code-block:: sh

    python make.py gamma-facet tox default
    flit publish

Please note the following:

*   Because the PyPI package index is immutable, it is recommended to do a test
    upload to `PyPI test <https://test.pypi.org/>`__ first. Ensure all metadata presents
    correctly before proceeding to proper publishing. The command to publish to test is

    .. code-block:: sh

        flit publish --repository testpypi

    which requires the specification of testpypi in a special ``.pypirc`` file
    with specifications as demonstrated `here
    <https://flit.readthedocs.io/en/latest/upload.html>`__.
*   The ``pyproject.toml`` does not provide specification for a short description
    (displayed in the top gray band on the PyPI page for the package). This description
    comes from the ``src/__init__.py`` script.
*   `flit <https://flit.readthedocs.io/en/latest/>`__ which is used here to publish to
    PyPI, also has the flexibility to support package building (wheel/sdist) via
    ``flit build`` and installing the package by copy or symlink via ``flit install``.
*   Build output will be stored in the ``dist/`` directory.

Conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

conda build metadata, build settings and package dependencies
are obtained from ``meta.yml``. To build and then publish the package to conda,
use the following commands:

.. code-block:: sh

    python make.py gamma-facet conda default
    anaconda upload --user BCG_Gamma dist/conda/noarch/<*package.tar.gz*>

Please note the following:

- Build output will be stored in the ``dist/`` directory.
- Some useful references for conda builds:

    - `Conda build tutorial
      <https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/building-conda-packages.html>`_
    - `Conda build metadata reference
      <https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html>`_

Azure DevOps CI/CD
--------------------

This project uses `Azure DevOps <https://dev.azure.com/>`_ for CI/CD pipelines.
The pipelines are defined in the ``azure-pipelines.yml`` file and are divided into
the following stages:

*   **code_quality_checks**: perform code quality checks for isort, black and flake8.
*   **detect_build_config_changes**: detect whether the build configuration as specified
    in the ``pyproject.yml`` has been modified. If it has, then a build test is run.
*   **Unit tests**: runs all unit tests and then publishes test results and coverage.
*   **conda_tox_build**: build the PyPI and conda distribution artifacts.
*   **Release**: see release process below for more detail.
*   **Docs**: build and publish documentation to GitHub Pages.

Release process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before initiating the release process, please ensure the version number
in ``src/__init__.py`` is correct and the format conforms to semantic
versioning. If the version needs to be corrected/bumped then open a PR for the
change and merge into develop before going any further.

The release process has the following key steps:

* Create a new release branch from develop and open a PR to master.
* Opening the PR to master will automatically run all conda/pip build tests via
  Azure Pipelines, triggering automatic upload of artifacts (conda and pip
  packages) to Azure DevOps. At this stage, it is recommended that the pip package
  build is checked using `PyPI test <https://test.pypi.org/>`__ to ensure all
  metadata presents correctly. This is important as package versions in
  PyPI proper are immutable.
* If everything passes and looks okay, merge the PR into master, this will
  trigger the release pipeline which will:

  * Tag the release commit with version number as specified in ``src/__init__.py``.
  * Create a release on GitHub for the new version, please check the `documentation 
    <https://docs.github.com/en/free-pro-team@latest/github/administering-a-repository/releasing-projects-on-github>`__
    for details.
  * Pre-fill the GitHub release title and description, including the changelog based on 
    commits since the last release. Please note this can be manually edited to be more
    succinct afterwards.
  * Attach build artifacts (conda and pip packages) to GitHub release.

*  Manually upload build artifacts to conda/PyPI using ``anaconda upload`` and
   ``flit publish``, respectively (see relevant sections under Package builds above)
   This may be automated in the future.
*  Remove any test versions for pip from PyPI test.
*  Merge any changes from release branch also back to develop.
*  Bump up version in ``src/__init__.py`` on develop to start work towards next release.
