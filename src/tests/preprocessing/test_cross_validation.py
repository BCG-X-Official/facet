import pytest
import pandas as pd
import numpy as np

# note: below is needed as a fixture
from tests.shared_fixtures import test_sample

def test_circular_cv_init(test_sample):
    # filter out warnings triggerd by sk-learn/numpy
    import warnings

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    from yieldengine.preprocessing.cross_validation import CircularCrossValidator

    # check erroneous inputs
    #   - test_ratio = 0
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(num_samples=100, test_ratio=0.0)

    #   - test_ratio < 0
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(num_samples=100, test_ratio=-0.1)

    #   - test_ratio > 1
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(num_samples=100, test_ratio=1.00001)

    #   - 0 samples per fold
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(num_samples=len(test_sample), test_ratio=0.00001)

    #   - (0 samples)
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(num_samples=0)

    #   - (#samples not specified)
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator()


def test_get_train_test_splits_as_dataframe(test_sample):
    from yieldengine.preprocessing.cross_validation import CircularCrossValidator

    my_ds = CircularCrossValidator(
        num_samples=len(test_sample), test_ratio=0.2, num_folds=50
    )

    list_of_train_test_splits = list(
        my_ds.get_train_test_splits_as_dataframes(input_dataset=test_sample)
    )

    # assert we get 50 folds
    assert len(list_of_train_test_splits) == 50

    # check correct ratio of test/train
    for train_set, test_set in list_of_train_test_splits:
        assert 0.19 < float(len(test_set) / len(test_sample) < 0.21)

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
        assert test_sample.equals(combined)


def test_get_train_test_splits_as_indices():
    from yieldengine.preprocessing.cross_validation import CircularCrossValidator

    test_folds = 200
    for use_bootstrapping in (False, True):

        my_cv = CircularCrossValidator(
            num_samples=123456,
            test_ratio=0.2,
            num_folds=test_folds,
            use_bootstrapping=use_bootstrapping,
        )

        list_of_train_test_splits = list(my_cv.get_train_test_splits_as_indices())

        # assert we get right amount of folds
        assert len(list_of_train_test_splits) == test_folds

        # check correct ratio of test/train
        for train_set, test_set in list_of_train_test_splits:
            assert 0.19 < float(len(test_set) / (len(test_set) + len(train_set)) < 0.21)

        list_of_train_test_splits_2 = list(my_cv.get_train_test_splits_as_indices())

        assert len(list_of_train_test_splits) == len(
            list_of_train_test_splits_2
        ), "The number of folds should be stable!"

        for f1, f2 in zip(list_of_train_test_splits, list_of_train_test_splits_2):
            assert np.array_equal(f1[0], f2[0]), "Fold indices should be stable!"
            assert np.array_equal(f1[1], f2[1]), "Fold indices should be stable!"

        if use_bootstrapping:
            # now test: resample() should change fold indices on the next call of get_train_test_splits...
            my_cv.resample()

            list_of_train_test_splits_3 = list(my_cv.get_train_test_splits_as_indices())

            # due to randomness, we need to check this with a threshold
            # we allow 2 folds to be randomly the same
            randomly_same_allowed_threshold = 2
            num_different_folds = 0
            for f1, f2 in zip(list_of_train_test_splits, list_of_train_test_splits_3):
                if not np.array_equal(f1[0], f2[0]) and not np.array_equal(
                    f1[1], f2[1]
                ):
                    num_different_folds = num_different_folds + 1

            assert num_different_folds >= (
                test_folds - randomly_same_allowed_threshold
            ), "There are too many equal folds!"


def test_circular_cv_with_sk_learn():
    # filter out warnings triggerd by sk-learn/numpy
    import warnings

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

    from sklearn import svm, datasets, tree
    from sklearn.model_selection import GridSearchCV
    from yieldengine.preprocessing.cross_validation import CircularCrossValidator

    # load example data
    iris = datasets.load_iris()

    # define a yield-engine circular CV:
    my_cv = CircularCrossValidator(
        num_samples=len(iris.data), test_ratio=0.21, num_folds=50
    )

    # define parameters and model
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC(gamma="scale")

    # use the defined my_cv circular CV within GridSearchCV:
    clf = GridSearchCV(svc, parameters, cv=my_cv)
    clf.fit(iris.data, iris.target)

    # test if the number of received folds is correct:
    assert clf.n_splits_ == 50, "50 folds should have been generated by the circular CV"

    assert clf.best_score_ > 0.85, "Expected a minimum score of 0.85"

    # define new paramters and a different model
    # use the defined my_cv circular CV again within GridSeachCV:
    parameters = {
        "criterion": ("gini", "entropy"),
        "max_features": ["sqrt", "auto", "log2"],
    }
    cl2 = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=my_cv)
    cl2.fit(iris.data, iris.target)

    assert cl2.best_score_ > 0.85, "Expected a minimum score of 0.85"


def test_duplicate_fold_warning(test_sample):
    from yieldengine.preprocessing.cross_validation import CircularCrossValidator

    with pytest.warns(expected_warning=UserWarning):
        # the 6th fold will be a duplicate, hence we expect a warning:
        my_cs = CircularCrossValidator(
            num_samples=100, test_ratio=0.2, num_folds=6, use_bootstrapping=False
        )
