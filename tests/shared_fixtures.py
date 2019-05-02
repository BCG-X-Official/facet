import os
import pandas as pd
import pytest
import yieldengine.core


@pytest.fixture
def test_sample():
    import tests.testdata

    data_folder_path = os.path.dirname(tests.testdata.__file__)

    # Note: this file is not included within the git repository!
    testdata_file_path = os.path.join(
        data_folder_path, "master_table_clean_anon_144.csv"
    )

    inputfile_config = yieldengine.core.get_global_config(section="inputfile")
    return pd.read_csv(
        filepath_or_buffer=testdata_file_path,
        delimiter=inputfile_config["delimiter"],
        header=inputfile_config["header"],
    )