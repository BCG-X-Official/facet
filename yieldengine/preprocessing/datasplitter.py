import pandas as pd
import numpy as np


def get_train_test_splits(
    input_dataset: pd.DataFrame, test_ratio: float = 0.2, num_folds: int = 50
):
    num_samples = len(input_dataset)
    num_test_samples_per_fold = int(num_samples * test_ratio)

    test_splits_start_samples = np.random.randint(0, num_samples - 1, num_folds)
    data_indices = np.arange(num_samples)

    for fold_test_start_sample in test_splits_start_samples:
        data_indices_rolled = np.roll(data_indices, fold_test_start_sample)
        test_indices = data_indices_rolled[0:num_test_samples_per_fold]
        train_indices = data_indices_rolled[num_test_samples_per_fold:]
        yield ((input_dataset.iloc[train_indices], input_dataset.iloc[test_indices]))
