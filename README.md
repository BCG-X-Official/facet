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

### 2.1 General guidelines

- The documentation is generated from docstrings in the source code

- Before writing your own documentation, take some time to study the documentation of
 the existing code, and try to emulate the same style
 
 - As a general rule, aim to describe not only _what_ the code does, but also _why_
, including the rationale for any design choices that may not be obvious

- Provide examples wherever this helps explain usage patterns


### 2.2 Docstring guidelines

- A docstring is mandatory for all of the following entities in the source code
, except when they are protected/private (i.e. the name starts with a leading `_
` character):
    - modules
    - classes
    - functions/methods
    - properties
    - atributes

- (From PEP 8): Docstrings are not necessary for non-public methods, but you should 
have a comment that describes what the method does.

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
        """
        Explains the inner workings of a predictive model based on the SHAP approach.
  
        The model inspector offers the following anaalyses:
        - ...
        - ...
    ```
    but not
    ```
    class ModelInspector:
        """
        This is a class that provides the functionality to inspect models
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
            """:return: the foo object"""
            pass
    ```



- Start full sentences and phrases with a capitalised word and end each sentence with
 punctuation
, e.g.,

    ```"""Fit the model."""```   

    but not   

    ```"""fit the model"""```

- Single-phrase parameter and property descriptions should not be capitalised and should
 not end with punctuation: 
 
     ```
    :param color_fg: the foreground color
    ```   

    but not   

     ```
    :param color_fg: The foreground color.
    ```   

- One-liner docstrings may be fitted on a single line, e.g.,

    ```"""Fit the model."""```

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

### 2.3 Docstring tips & tricks

- To render text as a code, enclose it in double backquotes, e.g.,
    ``` 
    ``print("hello")``
    ``` 
 
- To cross-reference objects in the docstrings use reStructured roles:
   - class: ```
            :class:`package.module.class_name`
            ```
   - method of a class/object: ```
                               :meth:`package.module.class.method`
                               ```
   - an attribute (and a property): ```
                                     :attr:`package.module.class.attribute`
                                     ```  
   - using instead ```
                   :attr:`~package.module.class.attribute`
                                     ```
     would display only the last part of the link `attribute` instead of the full 
     part  `package.module.class.attribute`.
   - if an identifier is defined in the current module, the cross-reference does not
    need to include the module path, as in
    ```
    :class:`ClassA`
    ```
    or in
    ```
    :attr:`ClassA.attribute_b`
    ``` 
          
- External links. To create a link to `https://scikit-learn.org/stable/`, labeled
 "scikit_learn", write
    ``` 
    `scikit-learn <https://scikit-learn.org/stable/>`_
    ```

     
