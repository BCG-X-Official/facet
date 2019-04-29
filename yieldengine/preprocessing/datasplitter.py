import pandas as pd
import numpy as np
from typing import Generator, Tuple


def get_train_test_splits(
    input_dataset: pd.DataFrame, test_ratio: float = 0.2, num_folds: int = 50
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """

    :param input_dataset:
    :param test_ratio:
    :param num_folds:
    :return:
    """
    if not type(input_dataset) == pd.DataFrame:
        raise ValueError("Expected a pandas.DataFrame as input_dataset")

    if not (0 < test_ratio < 1):
        raise ValueError("Expected (0 < test_ratio < 1), but %d was given" % test_ratio)

    num_samples = len(input_dataset)

    num_test_samples_per_fold = int(num_samples * test_ratio)

    if num_test_samples_per_fold == 0:
        raise ValueError(
            "The number of test samples per fold is 0 - increase ratio or size of input dataset"
        )

    test_splits_start_samples = np.random.randint(0, num_samples - 1, num_folds)
    data_indices = np.arange(num_samples)

    for fold_test_start_sample in test_splits_start_samples:
        data_indices_rolled = np.roll(data_indices, fold_test_start_sample)
        test_indices = data_indices_rolled[0:num_test_samples_per_fold]
        train_indices = data_indices_rolled[num_test_samples_per_fold:]
        yield ((input_dataset.iloc[train_indices], input_dataset.iloc[test_indices]))
