from gamma import Sample
from gamma.sklearndf import TransformerDF


def test_column_transformer_df(
    sample: Sample, simple_preprocessor: TransformerDF
) -> None:
    simple_preprocessor.fit_transform(X=sample.features)
