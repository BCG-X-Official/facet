import os
import pytest


@pytest.fixture
def data_file_path():
    import tests.testdata

    data_folder_path = os.path.dirname(tests.testdata.__file__)

    # Note: this file is not included within the git repository!
    testdata_file_path = os.path.join(
        data_folder_path, "master_table_clean_anon_144.csv"
    )

    return testdata_file_path


def test_load_raw_data(data_file_path):
    from yieldengine.loading import dataloader

    dataloader.load_raw_data(input_path=data_file_path)


def test_validate_raw_data(data_file_path):
    from yieldengine.loading import dataloader
    import yieldengine.core

    raw_data_df = dataloader.load_raw_data(input_path=data_file_path)
    dataloader.validate_raw_data(input_data_df=raw_data_df)

    # now test if the exception is properly thrown, when a column is missing:
    # load how the needed columns should be called:
    inputfile_config = yieldengine.core.get_global_config(section="inputfile")
    date_column_name = inputfile_config["date_column_name"]
    yield_column_name = inputfile_config["yield_column_name"]
    # remove them...
    raw_data_df = raw_data_df.drop(columns=[date_column_name, yield_column_name])

    # call validate again and expect an exception
    with pytest.raises(expected_exception=ValueError):
        dataloader.validate_raw_data(input_data_df=raw_data_df)
