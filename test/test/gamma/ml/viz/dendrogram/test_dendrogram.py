import pandas as pd
import pytest

from gamma.ml import Sample
from gamma.ml.fitcv import RegressorFitCV
from gamma.ml.inspection import RegressorInspector
from gamma.ml.validation import CircularCrossValidator
from gamma.ml.viz import DendrogramDrawer, DendrogramReportStyle
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF


@pytest.fixture()
def model_inspector(
    batch_table: pd.DataFrame, sample: Sample, simple_preprocessor: TransformerDF
) -> RegressorInspector:

    cv = CircularCrossValidator(test_ratio=0.20, num_splits=5)
    pipeline = RegressorPipelineDF(
        regressor=LGBMRegressorDF(), preprocessing=simple_preprocessor
    )
    return RegressorInspector(
        models=RegressorFitCV(pipeline=pipeline, cv=cv, sample=sample)
    )


def test_dendrogram_drawer_textstyle(model_inspector: RegressorInspector) -> None:
    linkage = model_inspector.cluster_dependent_features()
    dd = DendrogramDrawer(
        title="Test", linkage_tree=linkage, style=DendrogramReportStyle()
    )
    dd.draw()
