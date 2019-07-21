"""Plot facilities for EDA."""

import logging
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
COLOR_NON_OUTLIER = "blue"
COLOR_FAR_OUTLIER = "red"
COLOR_OUTLIER = "orange"

IQR_MULTIPLE = 1.5
IQR_MULTIPLE_FAR = None


def ecdf(
    data: pd.Series,
    iqr_multiple: Optional[float] = IQR_MULTIPLE,
    iqr_multiple_far: Optional[float] = IQR_MULTIPLE_FAR,
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float]
]:
    """Compute ECDF for a one-dimensional iterable of values.

    Return the x and y values of an empirical cumulative distribution plot of the
    values in `data`. Outlier and far outlier points are returned in separate lists.

    A sample is considered an outlier if it is outside the range
    :math:`[Q_1 - iqr\\_ multiple(Q_3-Q_1), Q_3 + iqr\\_ multiple(Q_3-Q_1)]`
    where :math:`Q_1` and :math:`Q_3` are the lower and upper quartiles. The same
    is used for far outliers with `iqr_multiple` replaced by `iqr_multiple_far`.

    :param data: the series of values forming our sample
    :param iqr_multiple: iqr multiple to determine outliers. If None then no
      outliers and far outliers are computed. Default is 1.5
    :param iqr_multiple_far: iqr multiple to determine far outliers. If None no far
      outliers are computed. Should be greater then iqr_multiple when both are not None
    :return: x_inlier, y_inlier, x_outlier, y_outlier, x_far_outlier, y_far_outlier
     the lists of x and y coordinates for the ecdf plot for the inlier, outlier and
     far outlier points.
    """
    if iqr_multiple and iqr_multiple_far:
        try:
            if iqr_multiple_far <= iqr_multiple:
                log.warning(
                    f"iqr_multiple={iqr_multiple} should be smaller than "
                    f"iqr_multiple_far={iqr_multiple_far}"
                )
        except TypeError:
            log.warning("iqr_multiple and iqr_multiple_far should be float or None")
    # x-data for the ECDF: x
    x = np.sort(data[~data.isna()])

    # Number of data points: n
    n = len(x)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1)

    if iqr_multiple:
        # outliers
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        out_lower = q1 - iqr_multiple * iqr
        out_upper = q3 + iqr_multiple * iqr
        inlier_mask = (x >= out_lower) & (x <= out_upper)

        if iqr_multiple_far:
            far_out_lower = q1 - iqr_multiple_far * iqr
            far_out_upper = q3 + iqr_multiple_far * iqr
            outlier_mask = (~inlier_mask) & (x >= far_out_lower) & (x <= far_out_upper)
            far_out_mask = ~(inlier_mask | outlier_mask)
        else:
            outlier_mask = ~inlier_mask
            far_out_mask = []

    else:
        inlier_mask = slice(None)
        outlier_mask = []
        far_out_mask = []

    return (
        x[inlier_mask],
        y[inlier_mask],
        x[outlier_mask],
        y[outlier_mask],
        x[far_out_mask],
        y[far_out_mask],
    )


def plot_ecdf(
    data: pd.Series,
    color_non_outlier: str = COLOR_NON_OUTLIER,
    color_outlier: str = COLOR_OUTLIER,
    color_far_outlier: str = COLOR_FAR_OUTLIER,
    iqr_multiple: float = IQR_MULTIPLE,
    iqr_multiple_far: float = IQR_MULTIPLE_FAR,
    **kwargs,
) -> None:
    """Plot an empirical cumulative distribution plot from a list-like data.

    Plot a scatter plot of the empirical cumulative distribution plot of the
    values in `data`. Outlier and far outlier points are shown in a different color.

    A sample is considered an outlier if it is outside the range
    :math:`[Q_1 - iqr\\_ multiple(Q_3-Q_1), Q_3 + iqr\\_ multiple(Q_3-Q_1)]`
    where :math:`Q_1` and :math:`Q_3` are the lower and upper quartiles. The same
    is used for far outliers with `iqr_multiple` replaced by `iqr_multiple_far`.

    :param data: the series of values forming our sample
    :param color_non_outlier: the color for non outlier points
    :param color_outlier: the color for outlier points
    :param color_far_outlier: the color for far outlier points
    :param iqr_multiple: iqr multiple to determine outliers. If None then no
      outliers and far outliers are computed. Default is 1.5
    :param iqr_multiple_far: iqr multiple to determine far outliers. If None no far
      outliers are computed. Should be greater then iqr_multiple when both are not None
    """
    x_inlier, y_inlier, x_outlier, y_outlier, x_far_outlier, y_far_outlier = ecdf(
        data, iqr_multiple=iqr_multiple, iqr_multiple_far=iqr_multiple_far
    )
    matplotlib_kwargs = {"marker": ".", "linestyle": "none"}
    matplotlib_kwargs.update(**kwargs)
    plt.plot(x_inlier, y_inlier, color=color_non_outlier, **matplotlib_kwargs)
    plt.plot(x_outlier, y_outlier, color=color_outlier, **matplotlib_kwargs)
    plt.plot(x_far_outlier, y_far_outlier, color=color_far_outlier, **matplotlib_kwargs)
    if hasattr(data, "name"):
        plt.xlabel(data.name)
    plt.ylabel("count")


def plot_ecdf_df(
    df: pd.DataFrame,
    color_non_outlier: str = COLOR_NON_OUTLIER,
    color_outlier: str = COLOR_OUTLIER,
    color_far_outlier: str = COLOR_FAR_OUTLIER,
    iqr_multiple=IQR_MULTIPLE,
    iqr_multiple_far=IQR_MULTIPLE_FAR,
) -> None:
    """Plot  empirical cumulative distributions of numerical columns of a dataframe.

    :param df: the dataframe whose numerical columns are used to compute ECDF's
    :param color_non_outlier: the color for non outlier points
    :param color_outlier: the color for outlier points
    :param color_far_outlier: the color for far outlier points
    :param iqr_multiple: iqr multiple to determine outliers. If None then no
      outliers and far outliers are computed. Default is 1.5
    :param iqr_multiple_far: iqr multiple to determine far outliers. If None no far
      outliers are computed. Should be greater then iqr_multiple when both are not None
    """
    numerical_features = df.select_dtypes(include="number").columns.values
    for feature in numerical_features:
        plot_ecdf(
            df.loc[:, feature],
            iqr_multiple=iqr_multiple,
            iqr_multiple_far=iqr_multiple_far,
            color_non_outlier=color_non_outlier,
            color_outlier=color_outlier,
            color_far_outlier=color_far_outlier,
        )
        plt.title(feature)
        plt.show()


def plot_hist_df(df: pd.DataFrame) -> None:
    """Plot the histograms of the categorical columns of a dataframe.

    :param df: the dataframe whose categorical columns are used for the count plot
    """
    categorical_features = df.select_dtypes(include=["category", object]).columns
    for feature in categorical_features:
        df.loc[:, feature].value_counts().plot(kind="bar")
        plt.title(feature)
        plt.show()
