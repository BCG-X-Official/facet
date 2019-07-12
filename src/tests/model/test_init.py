import pytest
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder

from tests.model import make_simple_transformer
from yieldengine.model import RegressionModel


def test_model() -> None:

    model = RegressionModel(
        estimator=LGBMRegressor(), preprocessing=make_simple_transformer()
    )

    # test-type check within constructor:
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        RegressionModel(estimator=LGBMRegressor(), preprocessing=OneHotEncoder())
