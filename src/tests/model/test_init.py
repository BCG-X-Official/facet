import pytest
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from tests.model import make_simple_transformer
from yieldengine.model import Model


def test_model() -> None:

    model = Model(predictor=LGBMRegressor(), preprocessing=make_simple_transformer())

    # test-type check within constructor:
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        Model(predictor=LGBMRegressor(), preprocessing=OneHotEncoder())
