"""
Plotting of distributions for exploratory data visualization

:meth:`plot_ecdf` plots empirical cumulative plots of numerical columns of a a list
like data (:class:`pandas.Series`, numerical list, etc).
"""

import logging
from typing import *

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.pyplot import gca

from gamma import ListLike

__all__ = ["plot_ecdf"]

log = logging.getLogger(__name__)

COLOR_NON_OUTLIER = "blue"
COLOR_FAR_OUTLIER = "purple"
COLOR_OUTLIER = "orange"

IQR_MULTIPLE = 1.5
IQR_MULTIPLE_FAR = 3.0


class _XYSeries(NamedTuple):
    """
    Series of x and y coordinates for plotting; x and y values are held in two
    separate lists of the same length.
    """

    x: ListLike[float]
    y: ListLike[float]


class _Ecdf(NamedTuple):
    """
    Three sets of coordinates for plotting an ECDF: inliers, outliers, and far
    outliers.
    """

    inliers: _XYSeries
    outliers: _XYSeries
    far_outliers: _XYSeries


def _ecdf(
    data: ListLike[float],
    iqr_multiple: Optional[float] = IQR_MULTIPLE,
    iqr_multiple_far: Optional[float] = IQR_MULTIPLE_FAR,
) -> _Ecdf:
    """
    Compute ECDF for scalar values.

    Return the x and y values of an empirical cumulative distribution plot of the
    values in ``data``. Outlier and far outlier points are returned in separate
    lists.

    A sample is considered an outlier if it is outside the range
    :math:`[Q_1 - iqr\\_ multiple(Q_3-Q_1), Q_3 + iqr\\_ multiple(Q_3-Q_1)]`
    where :math:`Q_1` and :math:`Q_3` are the lower and upper quartiles. The same
    is used for far outliers with ``iqr_multiple`` replaced by ``iqr_multiple_far``.

    :param data: the series of values forming our sample
    :param iqr_multiple: iqr multiple to determine outliers. If None then no
      outliers and far outliers are computed. Default is 1.5
    :param iqr_multiple_far: iqr multiple to determine far outliers. If None no far
      outliers are computed. Should be greater then iqr_multiple when both are not
      None. Default is 3.0
    :return: x_inlier, y_inlier, x_outlier, y_outlier, x_far_outlier, y_far_outlier
     the lists of x and y coordinates for the ecdf plot for the inlier, outlier and
     far outlier points.
    """

    if iqr_multiple and iqr_multiple_far:
        if iqr_multiple_far <= iqr_multiple:
            log.warning(
                f"arg iqr_multiple={iqr_multiple} must be smaller than "
                f"arg iqr_multiple_far={iqr_multiple_far}"
            )

    # x-data for the ECDF: x
    if isinstance(data, pd.Series):
        data = data.values
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    x = np.sort(data[~np.isnan(data)])

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

    return _Ecdf(
        _XYSeries(x[inlier_mask], y[inlier_mask]),
        _XYSeries(x[outlier_mask], y[outlier_mask]),
        _XYSeries(x[far_out_mask], y[far_out_mask]),
    )


def plot_ecdf(
    data: ListLike[float],
    ax: Optional[Axes] = None,
    iqr_multiple: float = IQR_MULTIPLE,
    iqr_multiple_far: float = IQR_MULTIPLE_FAR,
    color_non_outlier: str = COLOR_NON_OUTLIER,
    color_outlier: str = COLOR_OUTLIER,
    color_far_outlier: str = COLOR_FAR_OUTLIER,
    **kwargs,
) -> None:
    """
    Plot an empirical cumulative distribution plot from scalar values.

    Plot a scatter plot of the empirical cumulative distribution plot of the
    values in `data`. Outlier and far outlier points are shown in a different color.

    A sample is considered an outlier if it is outside the range
    :math:`[Q_1 - iqr\\_ multiple(Q_3-Q_1), Q_3 + iqr\\_ multiple(Q_3-Q_1)]`
    where :math:`Q_1` and :math:`Q_3` are the lower and upper quartiles. The same
    is used for far outliers with `iqr_multiple` replaced by `iqr_multiple_far`.

    :param data: the data of values forming our sample; must be scalar
    :param ax: the axes to plot on; if ``None``, use pyplot's current axes
    :param iqr_multiple: iqr multiple to determine outliers. If None then no
      outliers and far outliers are computed (default: 1.5)
    :param iqr_multiple_far: iqr multiple to determine far outliers. If ``None`` no far
      outliers are computed. Should be greater then iqr_multiple when both are not
      ``None`` (default: 3.0)
    :param color_non_outlier: the color for non outlier points (default: blue)
    :param color_outlier: the color for outlier points (default: orange)
    :param color_far_outlier: the color for far outlier points (default: red)
    :param kwargs: keyword arguments to pass on to method
      :meth:`matplotlib.axes.Axes.plot`
    """

    if ax is None:
        ax = gca()

    e: _Ecdf = _ecdf(data, iqr_multiple=iqr_multiple, iqr_multiple_far=iqr_multiple_far)
    matplotlib_kwargs = {"marker": ".", "linestyle": "none", **kwargs}
    ax.plot(e.inliers.x, e.inliers.y, color=color_non_outlier, **matplotlib_kwargs)
    ax.plot(e.outliers.x, e.outliers.y, color=color_outlier, **matplotlib_kwargs)
    ax.plot(
        e.far_outliers.x, e.far_outliers.y, color=color_far_outlier, **matplotlib_kwargs
    )

    # add plot title and labels
    if hasattr(data, "name"):
        ax.set_title(f"ECDF: {data.name}")
        ax.set_xlabel(data.name)
    else:
        ax.set_title(f"ECDF")
        ax.set_xlabel("value")
    ax.set_ylabel("count")
