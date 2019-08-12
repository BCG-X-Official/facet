import pandas as pd
import pytest
from matplotlib.pyplot import figure

from gamma.ml import Sample
from gamma.ml.fitcv import RegressorFitCV
from gamma.ml.inspection import RegressionModelInspector
from gamma.ml.validation import CircularCrossValidator
from gamma.ml.viz import DendrogramDrawer, DendrogramFeatMapStyle, DendrogramLineStyle
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressionPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF


@pytest.fixture()
def model_inspector(
    batch_table: pd.DataFrame, sample: Sample, simple_preprocessor: TransformerDF
) -> RegressionModelInspector:

    cv = CircularCrossValidator(test_ratio=0.20, num_splits=5)
    pipeline = RegressionPipelineDF(
        regressor=LGBMRegressorDF(), preprocessing=simple_preprocessor
    )
    return RegressionModelInspector(
        models=RegressorFitCV(pipeline=pipeline, cv=cv, sample=sample)
    )


def test_linkage_drawer_style(model_inspector: RegressionModelInspector) -> None:
    linkage = model_inspector.cluster_dependent_features()
    fig = figure(figsize=(8, 16))
    ax = fig.add_subplot(111)
    dd = DendrogramDrawer(
        title="Test", linkage_tree=linkage, style=DendrogramLineStyle(ax=ax)
    )
    dd.draw()
    dd_2 = DendrogramDrawer(
        title="Test", linkage_tree=linkage, style=DendrogramFeatMapStyle(ax=ax)
    )
    dd_2.draw()
