import pytest
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from gamma.sklearndf.pipeline import RegressorPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF
from test.gamma.ml import make_simple_transformer


def test_model() -> None:

    model = RegressorPipelineDF(
        regressor=LGBMRegressorDF(), preprocessing=make_simple_transformer()
    )

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        RegressorPipelineDF(regressor=LGBMRegressor(), preprocessing=OneHotEncoder())
