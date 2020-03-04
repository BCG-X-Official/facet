# gamma-ml

Currently this project contains:

- `gamma.ml` the main package for various Machine Learning tasks like model 
selection, validation, fitting & predicting as well as inspection (using SHAP).
- `gamma.ml.Sample` utility class wrapping around a Pandas dataframe, easing common
ML related operations
- `gamma.ml.viz` Drawer and styles for Dendrograms, OOP model of a scipy linkage matrix

# Installation
Latest stable conda package `gamma-ml` can be installed using:
`conda install -c https://machine-1511619-alpha:bcggamma2019@artifactory.gamma.bcg.com/artifactory/api/conda/local-conda-1511619-alpha-01 gamma-ml`

Or add the alpha channel and this package to your `environment.yml`:
```
channels:
  - conda-forge
  - https://machine-1511619-alpha:bcggamma2019@artifactory.gamma.bcg.com/artifactory/api/conda/local-conda-1511619-alpha-01
dependencies:
  - gamma-ml
```
# Documentation
Documentation for all of alpha's Python projects is available at: 
https://git.sourceai.io/pages/alpha/alpha/

# API-Reference
See: https://git.sourceai.io/pages/alpha/alpha/gamma.ml.html

# Contribute & Develop
Check out https://git.sourceai.io/alpha/alpha for developer instructions and guidelines.