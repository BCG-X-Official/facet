import pytest

import tests
from tests.loading import dataloader
from tests.paths import TEST_DATA_CSV


def test_load_raw_data() -> None:
    dataloader.load_raw_data(input_path=TEST_DATA_CSV)


def test_validate_raw_data() -> None:
    raw_data_df = dataloader.load_raw_data(input_path=TEST_DATA_CSV)
    dataloader.validate_raw_data(input_data_df=raw_data_df)

    # now test if the exception is properly thrown, when a column is missing:
    # load how the needed columns should be called:
    inputfile_config = tests.read_test_config(section="inputfile")
    date_column_name = inputfile_config["date_column_name"]
    yield_column_name = inputfile_config["yield_column_name"]
    # remove them...
    raw_data_df = raw_data_df.drop(columns=[date_column_name, yield_column_name])

    # call validate again and expect an exception
    with pytest.raises(expected_exception=ValueError):
        dataloader.validate_raw_data(input_data_df=raw_data_df)
