import pandas as pd

# noinspection PyPackageRequirements
import pytest

from gamma.ml import Sample
from gamma.ml.crossfit import RegressorCrossfit
from gamma.ml.inspection import RegressorInspector
from gamma.ml.validation import BootstrapCV
from gamma.ml.viz import DendrogramDrawer, DendrogramReportStyle
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF


@pytest.fixture()
def model_inspector(
    batch_table: pd.DataFrame, sample: Sample, simple_preprocessor: TransformerDF
) -> RegressorInspector:

    cv = BootstrapCV(random_state=42)

    pipeline = RegressorPipelineDF(
        regressor=LGBMRegressorDF(), preprocessing=simple_preprocessor
    )

    regressor_inspector = RegressorInspector(
        crossfit=RegressorCrossfit(base_estimator=pipeline, cv=cv).fit(sample=sample)
    )

    return regressor_inspector


def test_dendrogram_drawer_text(model_inspector: RegressorInspector) -> None:
    linkage = model_inspector.cluster_dependent_features()
    dd = DendrogramDrawer(title="Test", linkage=linkage, style=DendrogramReportStyle())
    dd.draw()
