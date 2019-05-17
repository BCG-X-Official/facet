from yieldengine import Sample
from yieldengine.feature.transform import DataFrameTransformer


def test_column_transformer_df(
    sample: Sample, df_transformer: DataFrameTransformer
) -> None:
    df_transformer.fit_transform(X=sample.features)
