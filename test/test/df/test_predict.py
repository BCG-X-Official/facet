import numpy as np
import pandas as pd

from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.regression import RandomForestRegressorDF


def test_dataframe_predictor_classifier(
    iris_df: pd.DataFrame, iris_target: str
) -> None:
    # implement a lightweight BasePredictorWrapperDF for RandomForestClassifier...

    classifier_df = RandomForestClassifierDF()

    x = iris_df.drop(columns=iris_target)
    y = iris_df.loc[:, iris_target]

    classifier_df.fit(X=x, y=y)

    predictions = classifier_df.predict(X=x)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y)
    assert np.all(predictions.isin(y.unique()))

    # test predict_proba & predict_log_proba
    for func in (classifier_df.predict_proba, classifier_df.predict_log_proba):
        predicted_probas = func(X=x)

        # test data-type and shape
        assert isinstance(predicted_probas, pd.DataFrame)
        assert len(predicted_probas) == len(y)
        assert predicted_probas.shape == (len(y), len(y.unique()))

        # check correct labels are set as columns
        assert list(y.unique()) == list(predicted_probas.columns)


def test_dataframe_predictor_regressor(
    boston_df: pd.DataFrame, boston_target: str
) -> None:
    # implement a lightweight RegressorWrapperDF for RandomForestRegressor...

    classifier_df = RandomForestRegressorDF()

    x = boston_df.drop(columns=boston_target)
    y = boston_df.loc[:, boston_target]

    classifier_df.fit(X=x, y=y)

    predictions = classifier_df.predict(X=x)

    # test predictions data-type, length and values
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y)
