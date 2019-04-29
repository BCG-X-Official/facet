import pytest
import pandas as pd
import os
import yieldengine.core


@pytest.fixture
def test_dataframe():
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


def test_get_train_test_splits(test_dataframe: pd.DataFrame):
    from yieldengine.preprocessing import datasplitter

    list_of_train_test_splits = list(
        datasplitter.get_train_test_splits(
            input_dataset=test_dataframe, test_ratio=0.2, num_folds=50
        )
    )

    # assert we get 50 folds
    assert len(list_of_train_test_splits) == 50

    # check correct ratio of test/train
    for train_set, test_set in list_of_train_test_splits:
        assert 0.19 < float(len(test_set) / len(test_dataframe) < 0.21)

    # check all generated folds
    for train_set, test_set in list_of_train_test_splits:
        # assert test/train are mutually exclusive
        assert (
            len(
                train_set.merge(
                    right=test_set, how="inner", left_index=True, right_index=True
                )
            )
            == 0
        )
        # assert test/train add up back to the complete dataset
        combined = pd.concat([train_set, test_set], axis=0).sort_index()
        assert test_dataframe.equals(combined)

    # check erroneous inputs
    #   - test_ratio = 0
    with pytest.raises(expected_exception=ValueError):
        list(
            datasplitter.get_train_test_splits(
                input_dataset=test_dataframe, test_ratio=0.0
            )
        )
    #   - test_ratio < 0
    with pytest.raises(expected_exception=ValueError):
        list(
            datasplitter.get_train_test_splits(
                input_dataset=test_dataframe, test_ratio=-0.1
            )
        )

    #   - test_ratio > 1
    with pytest.raises(expected_exception=ValueError):
        list(
            datasplitter.get_train_test_splits(
                input_dataset=test_dataframe, test_ratio=1.00001
            )
        )

    #   - 0 samples per fold
    with pytest.raises(expected_exception=ValueError):
        list(
            datasplitter.get_train_test_splits(
                input_dataset=test_dataframe, test_ratio=0.00001
            )
        )

    #   - empty input dataframe (0 samples)
    with pytest.raises(expected_exception=ValueError):
        list(datasplitter.get_train_test_splits(input_dataset=pd.DataFrame()))
