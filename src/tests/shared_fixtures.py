import pandas as pd
import pytest

from tests import read_test_config
from tests.paths import TEST_DATA_CSV


@pytest.fixture
def batch_table() -> pd.DataFrame:

    # Note: this file is not included within the git repository!
    inputfile_config = read_test_config(section="inputfile")
    return pd.read_csv(
        filepath_or_buffer=TEST_DATA_CSV,
        delimiter=inputfile_config["delimiter"],
        header=inputfile_config["header"],
        decimal=inputfile_config["decimal"],
    )
