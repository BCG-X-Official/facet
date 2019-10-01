import pandas as pd

# noinspection PyPackageRequirements
import pytest

from gamma.ml import Sample
from gamma.ml.predictioncv import RegressorPredictionCV
from gamma.ml.inspection import RegressorInspector
from gamma.ml.validation import CircularCV
from gamma.ml.viz import DendrogramDrawer, DendrogramReportStyle
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF


@pytest.fixture()
def model_inspector(
    batch_table: pd.DataFrame, sample: Sample, simple_preprocessor: TransformerDF
) -> RegressorInspector:

    cv = CircularCV(test_ratio=0.20, n_splits=5)
    pipeline = RegressorPipelineDF(
        regressor=LGBMRegressorDF(), preprocessing=simple_preprocessor
    )
    return RegressorInspector(
        predictions=RegressorPredictionCV(pipeline=pipeline, cv=cv, sample=sample)
    )


def test_dendrogram_drawer_textstyle(model_inspector: RegressorInspector) -> None:
    linkage = model_inspector.cluster_dependent_features()
    dd = DendrogramDrawer(
        title="Test", linkage_tree=linkage, style=DendrogramReportStyle()
    )
    dd.draw()
