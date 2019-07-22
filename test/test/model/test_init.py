import pytest
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from gamma.model import ModelPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF
from test.model import make_simple_transformer


def test_model() -> None:

    model = ModelPipelineDF(
        predictor=LGBMRegressorDF(), preprocessing=make_simple_transformer()
    )

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        ModelPipelineDF(predictor=LGBMRegressor(), preprocessing=OneHotEncoder())
