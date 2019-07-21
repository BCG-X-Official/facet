from yieldengine import Sample
from yieldengine.df.transform import DataFrameTransformerWrapper


def test_column_transformer_df(
    sample: Sample, simple_preprocessor: DataFrameTransformerWrapper
) -> None:
    simple_preprocessor.fit_transform(X=sample.features)
