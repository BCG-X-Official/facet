import logging
import warnings

import pytest

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", "You can install the OpenMP library")


# noinspection PyMissingTypeHints
@pytest.fixture
def LGBMRegressor():
    warnings.filterwarnings(
        "ignore", message=r"Starting from version 2\.2\.1", category=UserWarning
    )

    from lightgbm.sklearn import LGBMRegressor

    return LGBMRegressor
