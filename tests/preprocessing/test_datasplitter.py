import pytest
import pandas as pd
import numpy as np
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


def test_datasplitter_init(test_dataframe: pd.DataFrame):
    from yieldengine.preprocessing.data_splitter import DataSplitter

    # check erroneous inputs
    #   - test_ratio = 0
    with pytest.raises(expected_exception=ValueError):
        DataSplitter(num_samples=100, test_ratio=0.0)

    #   - test_ratio < 0
    with pytest.raises(expected_exception=ValueError):
        DataSplitter(num_samples=100, test_ratio=-0.1)

    #   - test_ratio > 1
    with pytest.raises(expected_exception=ValueError):
        DataSplitter(num_samples=100, test_ratio=1.00001)

    #   - 0 samples per fold
    with pytest.raises(expected_exception=ValueError):
        DataSplitter(num_samples=len(test_dataframe), test_ratio=0.00001)

    #   - (0 samples)
    with pytest.raises(expected_exception=ValueError):
        DataSplitter(num_samples=0)

    #   - (#samples not specified)
    with pytest.raises(expected_exception=ValueError):
        DataSplitter()


def test_get_train_test_splits_as_dataframe(test_dataframe: pd.DataFrame):
    from yieldengine.preprocessing.data_splitter import DataSplitter

    my_ds = DataSplitter(num_samples=len(test_dataframe), test_ratio=0.2, num_folds=50)

    list_of_train_test_splits = list(
        my_ds.get_train_test_splits_as_dataframes(input_dataset=test_dataframe)
    )

    # assert we get 50 folds
    assert len(list_of_train_test_splits) == 50

    # check correct ratio of test/train
    for train_set, test_set in list_of_train_test_splits:
        assert 0.19 < float(len(test_set) / len(test_dataframe) < 0.21)

    # check all generated folds
    for train_set, test_set in list_of_train_test_splits:
        # assert the correct datatype (pd.DataFrame) is returned
        assert (
            type(test_set) == pd.DataFrame
        ), "test_set should be of type pd.DataFrame!"

        assert (
            type(train_set) == pd.DataFrame
        ), "train_set should be of type pd.DataFrame!"

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


def test_get_train_test_splits_as_indices():
    from yieldengine.preprocessing.data_splitter import DataSplitter

    test_folds = 200

    my_ds = DataSplitter(num_samples=123456, test_ratio=0.2, num_folds=test_folds)

    list_of_train_test_splits = list(my_ds.get_train_test_splits_as_indices())

    list_of_train_test_splits_2 = list(my_ds.get_train_test_splits_as_indices())

    assert len(list_of_train_test_splits) == len(
        list_of_train_test_splits_2
    ), "The number of folds should be stable!"

    for f1, f2 in zip(list_of_train_test_splits, list_of_train_test_splits_2):
        assert np.array_equal(f1[0], f2[0]), "Fold indices should be stable!"
        assert np.array_equal(f1[1], f2[1]), "Fold indices should be stable!"

    # now test the opposite: resample() should change fold indices on the next call of get_train_test_splits...
    my_ds.resample()

    list_of_train_test_splits_3 = list(my_ds.get_train_test_splits_as_indices())

    # due to randomness, we need to check this with a threshold
    # we allow 2 folds to be randomly the same
    randomly_same_allowed_threshold = 2
    num_different_folds = 0
    for f1, f2 in zip(list_of_train_test_splits, list_of_train_test_splits_3):
        if not np.array_equal(f1[0], f2[0]) and not np.array_equal(f1[1], f2[1]):
            num_different_folds = num_different_folds + 1

    assert num_different_folds >= (
        test_folds - randomly_same_allowed_threshold
    ), "There are too many equal folds!"


def test_datasplitter_with_sk_learn():
    # filter out warnings triggerd by sk-learn/numpy
    import warnings

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    from sklearn import svm, datasets, tree
    from sklearn.model_selection import GridSearchCV
    from yieldengine.preprocessing.data_splitter import DataSplitter

    # load example data
    iris = datasets.load_iris()

    # define a yield-engine Datasplitter:
    my_ds = DataSplitter(num_samples=len(iris.data), test_ratio=0.2, num_folds=50)

    # define parameters and model
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC(gamma="scale")

    # use the defined my_ds Datasplitter within GridSearchCV:
    clf = GridSearchCV(svc, parameters, cv=my_ds.get_train_test_splits_as_indices())
    clf.fit(iris.data, iris.target)

    # test if the number of received folds is correct:
    assert (
        clf.n_splits_ == 50
    ), "50 folds should have been generated by the Datasplitter"

    assert clf.best_score_ > 0.85, "Expected a minimum score of 0.85"

    # define new paramters and a different model
    # use the defined my_ds Datasplitter again within GridSeachCV:
    parameters = {
        "criterion": ("gini", "entropy"),
        "max_features": ["sqrt", "auto", "log2"],
    }
    cl2 = GridSearchCV(
        tree.DecisionTreeClassifier(),
        parameters,
        cv=my_ds.get_train_test_splits_as_indices(),
    )
    cl2.fit(iris.data, iris.target)

    assert cl2.best_score_ > 0.85, "Expected a minimum score of 0.85"
