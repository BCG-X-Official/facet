import pytest
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from gamma.sklearndf.pipeline import RegressionPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF
from test.model import make_simple_transformer


def test_model() -> None:

    model = RegressionPipelineDF(
        regressor=LGBMRegressorDF(), preprocessing=make_simple_transformer()
    )

    # test-type check within constructor:
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        RegressionPipelineDF(regressor=LGBMRegressor(), preprocessing=OneHotEncoder())
