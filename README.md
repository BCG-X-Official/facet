# yield-engine

You can find the API reference of yield-engine at
[https://git.sourceai.io/pages/schneider-joerg/yield-engine/]().

## Table of Contents  

<!-- TOC depthFrom:2 -->

- [1. Setup](#1-setup)
    - [1.1 Python Environment](#11-python-environment)
    - [1.2 Pytest](#12-pytest)
    - [1.3 Sphinx Documentation](#13-sphinx-documentation)
- [2. Git guidelines](#2-git-guidelines)
- [3. Documentation guidelines](#3-documentation-guidelines)
  - [3.1 General guidelines](#31-general-guidlines)
  - [3.2 Docstring guidelines](#32-docstring-guidelines)
  - [3.3 Docstring tips and tricks](#33-docstring-tips--tricks)


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

## 2. Git guidelines

- For commits to GitHub, phrase commit comments as the completion of the sentence
 "This commit will <...>", e.g.
    ```
    add method foo to class Bar
    ``` 
    but not
    ```
    added method foo to class Bar 
    ```

## 3. Documentation guidelines

### 3.1 General guidelines

- The documentation is generated from docstrings in the source code

- Before writing your own documentation, take some time to study the documentation of
 the existing code, and try to emulate the same style
 
 - As a general rule, aim to describe not only _what_ the code does, but also _why_
, including the rationale for any design choices that may not be obvious

- Provide examples wherever this helps explain usage patterns


### 3.2 Docstring guidelines

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

- For method arguments, return value, and class parameters, one must hint the type 
using the typing module. Hence do not specify the parameter types in the docstrings 
(see [link](#sphinx-type-hint)), e.g. 
    ```
    def f(x: int) -> float:
       """
       Do something.
       
       :param x: input value
       :return: output value
    ```
    but not
    ```
    def f(x: int) -> float:
       """
       Do something.
       
       :param int x: input value
       :return float: output value
    ```    

### 3.3 Docstring tips & tricks

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
    - if an object is located in another module, you can prepend a dot to refer to it
        . For instance in module `package1.packag3.module1` and you want to refer to 
        `package1.package3.package4.module2.MyClass`  the following will work
        ``` 
        """We use :class:`.Myclass` in this module."""
        ```
        If there are several `MyClass` across the project, this is ambiguous and 
        sphinx raises an warning.
    - if an object has been imported in a module it won't be recognized by 
        cross-referencing directly, hence
        ``` 
        """This module uses :class:`SomeClass`"""
        from other_module import SomeClass
        ```
        won't work, but
        ``` 
        """This module uses :class:`.SomeClass`"""
        from other_module import SomeClass
        ```
        will work (see above explanation).
                  
- External links. To create a link to `https://scikit-learn.org/stable/`, labeled
    "scikit_learn", write
    ``` 
    `scikit-learn <https://scikit-learn.org/stable/>`_
    ```
    
- <a id="sphinx-type-hint"></a> The sphinx extension  [sphinx-autodoc-typehints]
    (https://github.com/agronholm/sphinx-autodoc-typehints) extract the type hinting and 
    infers the parameter types of methods and classes when generating the documentation 
    from docstrings. To install the extension:
    ```
    conda install -c conda-forge sphinx-autodoc-typehints=1.6
    ``` 
    For sphinx to use it the `extensions` variable in `conf.py` must contain 
    `"sphinx_autodoc_typehints"`. 
    
- The `intersphinx` extension allows to automatically link to other objects of other 
    sphinx projects. This works for `sklearn`, `numpy`, `matplotlib`, `shap`. For it to 
    work 
    the extensions attribute of `conf.py` must contain `sphinx.ext
    .intersphins` and one must specify an `intersphinx_mapping` mapping to url links 
    of the documentation. Caveat: ```:mod:`shap`.
    Intersphinx does not care look at the context of imported modules, e.g.
    ``` 
    import pandas as pd
    The :class:`pd.DataFrame` is useful
    ```
    will not generate a url likn, while
     ``` 
    import pandas as pd
    The :class:`pandas.DataFrame` is useful
    ```
    will generate a link although in the module `pandas` itself is not recognized.
    **Beware that intersphinx and sphinx-autodoc-typehints seem not to work together**.
    
     
