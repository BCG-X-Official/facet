#!/bin/sh
conda env create -f environment.yml
conda activate facet-develop
pre-commit install