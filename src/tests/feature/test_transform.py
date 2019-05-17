from yieldengine import Sample


def test_column_transformer_df(sample: Sample, transformer_step) -> None:
    transformer_step.transformer.fit_transform(X=sample.features)
