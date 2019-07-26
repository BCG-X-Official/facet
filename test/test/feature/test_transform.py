import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from gamma import Sample
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.pipeline import PipelineDF
from gamma.sklearndf.transformation import SimpleImputerDF
from gamma.sklearndf.transformation.extra import BorutaDF


def test_column_transformer_df(
    sample: Sample, simple_preprocessor: TransformerDF
) -> None:
    simple_preprocessor.fit_transform(X=sample.features)


def test_boruta_df(boston_df: pd.DataFrame, boston_target: str) -> None:
    # test a pipeline with on the boston dataset

    boruta_selector = PipelineDF(
        steps=[
            ("preprocess", SimpleImputerDF(strategy="median")),
            (
                "boruta",
                BorutaDF(
                    estimator=RandomForestRegressor(), n_estimators=10, max_iter=10
                ),
            ),
        ]
    )

    x = boston_df.drop(columns=boston_target)
    y = boston_df.loc[:, boston_target]

    boruta_selector.fit(x, y)
