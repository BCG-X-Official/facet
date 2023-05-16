"""
Test shap decomposition calculations
"""
import logging
from typing import Set

import numpy as np

from pytools.data import Matrix
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression.extra import LGBMRegressorDF

from facet.data import Sample
from facet.explanation import TreeExplainerFactory
from facet.inspection import LearnerInspector

log = logging.getLogger(__name__)


def test_feature_affinity_matrices(
    best_lgbm_model: RegressorPipelineDF[LGBMRegressorDF],
    sample: Sample,
    n_jobs: int,
) -> None:
    regressor_inspector: LearnerInspector[
        RegressorPipelineDF[LGBMRegressorDF]
    ] = LearnerInspector(
        model=best_lgbm_model,
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", uses_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(
        sample
    )

    preprocessed_feature_names: Set[str] = set(
        best_lgbm_model.final_estimator.feature_names_in_
    )

    # feature affinity matrices (feature dependencies)
    # check that dimensions of pairwise feature matrices are equal to # of features,
    # and value ranges:
    for matrix, matrix_name in zip(
        (
            regressor_inspector.feature_association_matrix(),
            regressor_inspector.feature_synergy_matrix(),
            regressor_inspector.feature_redundancy_matrix(),
        ),
        ("association", "synergy", "redundancy"),
    ):
        matrix_full_name = f"feature {matrix_name} matrix"
        n_features = len(preprocessed_feature_names)
        assert isinstance(matrix, Matrix)
        assert matrix.values.shape[0] == n_features, f"rows in {matrix_full_name}"
        assert matrix.values.shape[1] == n_features, f"columns in {matrix_full_name}"
        assert matrix.names[0] is not None
        assert (
            set(matrix.names[0]) == preprocessed_feature_names
        ), f"row names in {matrix_full_name}"
        assert matrix.names[1] is not None
        assert (
            set(matrix.names[1]) == preprocessed_feature_names
        ), f"column names in {matrix_full_name}"

        # check values
        assert (
            np.nanmin(matrix.values) >= 0.0 and np.nanmax(matrix.values) <= 1.0
        ), f"Values of [0.0, 1.0] in {matrix_full_name}"
