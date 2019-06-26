"""Module with plot facilities for EDA."""
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ecdf(data: Iterable):
    """Compute ECDF for a one-dimensional array of measurements."""
    # x-data for the ECDF: x
    x = np.sort(data[~data.isna()])

    # Number of data points: n
    n = len(x)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1)

    # outliers
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    out_lower = q1 - 1.5 * iqr
    out_upper = q3 + 1.5 * iqr
    far_out_lower = q1 - 3 * iqr
    far_out_upper = q3 + 3 * iqr

    inlier_mask = (x >= out_lower) & (x <= out_upper)
    outlier_mask = (~inlier_mask) & (x >= far_out_lower) & (x <= far_out_upper)
    far_out_mask = ~(inlier_mask | outlier_mask)

    return x[inlier_mask], y[inlier_mask], x[outlier_mask], y[outlier_mask], x[
        far_out_mask], y[far_out_mask]


def plot_ecdf(data: Iterable):
    """Plot an empirical cumulative distribution plot from a list-like data."""
    plt.plot(*ecdf(data), marker='.', linestyle='none')
    if hasattr(data, 'name'):
        plt.xlabel(data.name)
    plt.ylabel('count')


def plot_ecdf_df(df: pd.DataFrame):
    """Plot the empirical cumulative distributions of the numerical columns of a \
    dataframe."""
    numerical_features = df.select_dtypes(include='number').columns.values
    for feature in numerical_features:
        plot_ecdf(df.loc[:, feature])
        plt.title(feature)
        plt.show()

def plot_hist_df(df:pd.DataFrame):
    """Plot the histograms of the categorical columns of a dataframe."""
    categorical_features = df.select_dtypes(include=['category', object]).columns
    for feature in categorical_features:
        df.loc[:, feature].value_counts().plot(kind='bar')
        plt.title(feature)
        plt.show()
