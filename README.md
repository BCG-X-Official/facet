# yield-engine

You can find the API reference of yield-engine at
[https://git.sourceai.io/pages/schneider-joerg/yield-engine/]().

## Table of Contents  

<!-- TOC depthFrom:2 -->

- [1. Setup](#1-setup)
    - [1.1 Python Environment](#11-python-environment)
    - [1.2 Pytest](#12-pytest)
    - [1.3 Sphinx Documentation](#13-sphinx-documentation)

## 1. Setup

### 1.1 Python Environment
There is a `environment.yml` provided in the project root folder, which you can use with Anaconda to set up a virtualenv for yieldengine.

### 1.2 Pytest
Run `pytest tests/` from the project root folder (or use the PyCharm test runner).
To measure test coverage, use `pytest --cov=yieldengine
 tests/`.
 
### 1.3 Sphinx Documentation
The generated Sphinx documentation of yieldengine is located at _/docs_. 
To build the documentation, ensure you have the Python packages `sphinx=2.0.1` and `sphinx_rtd_theme=0.4.3` installed, which we have purposely not included into the `environment.yml`. 
To update, simply run `make html` from within _/sphinx_. 
By default `make html` only compiles files which have been modfified since last 
compilation. To force the compilation of the full documentation enter first `make 
clean`.  

## 2. Documentation guideline

- The documentation is generated from docstrings in the source code

- Before writing your own documentation, take some time to study the documentation of
 the existing code, and try to emulate the same style

- A docstring is mandatory for all of the following entities in the source code
, except when they are protected/private (i.e. the name starts with a leading `_
` character):
    - modules
    - classes
    - functions/methods
    - properties
    - atributes

- Docstrings must follow the reStructuredText syntax (i.e., the default syntax for
 Sphinx)

- Write docstrings for functions and methods in the imperative style, e.g.,
    ```
    def fit():
    """Fit the model."""
    ```
    but not   
    
    ```
    def fit():
        """This is a function that fits the model."""
    ```
    (too wordy and not imperative)

- write docstrings for modules, classes, modules, and attributes starting with a
 descriptive phrase (as you would expect in a dictionary entry). Be concise and avoid
  unnecessary or redundant phrases. For example:
    ```
    class ModelInspector:
        """Explains the inner workings of a predictive model based on the SHAP approach.
  
        The model inspector offers the following anaalyses:
        - ...
        - ...
    ```
    but not
    ```
    class ModelInspector:
        """This is a class that provides the functionality to inspect models
        ...
    ```
    (too verbose, and explains the class in terms of its name which does not add any
     information)
 
 
 - Properties should be documented as if they were attributes, not as methods, e.g.,
    ```
    @property
        def children(self) -> Foo:
            """the child nodes of the tree"""
            pass
    ```
    but not
    ```
    @property
        def foo(self) -> Foo:
            """
            :return: the foo object"""
            pass
    ```



- Start full sentences with a capitalised word and end each sentence with punctuation
, e.g.,

    ```"""Fit the model."""```   

    but not   

    ```"""fit the model"""```

- Phrases that are not full sentences should not be capitalised and should not end
 with punctuation: 
 
     ```
    :param color_fg: the foreground color
    ```   

    but not   

     ```
    :param color_fg: The foreground color.
    ```   

- Fit one-liner docstrings on a single line, e.g.,

    ```"""Fit the model."""```

    but not 

    ```
    """
    Fit the model.
    """
    ```

- For multi-line docstrings, insert a line break after the leading triple quote and
 before the trailing triple quote, e.g.,
    ```
    def fit():
        """
        Fit the model.
        
        Use the underlying estimator's `fit` method
        to fit the model using the given training sample.
        
        :param sample: training sample
        """
    ```
    but not
    ```
    def fit():
        """Fit the model.
        
        Use the underlying estimator's `fit` method
        to fit the model using the given training sample.
        
        :param sample: training sample"""
    ```


- As a general rule, aim to describe not only _what_ the code does, but also _why_
, including the rationale for any design choices that may not be obvious

- Provide examples wherever this helps explain usage patterns
 