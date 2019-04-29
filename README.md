# yield-engine

You can find the API reference of yield-engine [here](https://git.sourceai.io/pages/schneider-joerg/yield-engine/).

##### Table of Contents  

<!-- TOC depthFrom:2 -->

- [1. Setup](#1-setup)
    - [1.1 Python Environment](#11-python-environment)
    - [1.2 Pytest](#12-pytest)
    - [1.3 Sphinx Documentation](#13-sphinx-documentation)

## 1. Setup
### 1.1 Python Environment
There is a `environment.yml` provided in the project root folder, which you can use with Anaconda to set up a virtualenv for yieldengine.
### 1.2 Pytest
Simply run `pytest tests/` from the project root folder (or use the PyCharm testrunner). The execution of pytest with coverage can be triggered using `pytest --cov=yieldengine tests/`.
### 1.3 Sphinx Documentation
The generated Sphinx documentation of yieldengine is located at _/docs_. To build the documentation, ensure you have the Python packages `sphinx=2.0.1` and `sphinx_rtd_theme=0.4.3` installed, which we have purposely not included into the `environment.yml`. To update, simply run `make html` from within _/sphinx_. **Note: the Makefile (-> make.bat) has not been adapted/tested for Windows**
