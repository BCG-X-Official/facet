#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
MVC-based classes for drawing charts.
"""
import logging
import sys
from abc import ABC, abstractmethod
from typing import *
from typing import TextIO

import matplotlib.pyplot as plt
from matplotlib import text as mt
from matplotlib.axes import Axes

log = logging.getLogger(__name__)

T_Model = TypeVar("T_Model")
T_Style = TypeVar("T_Style", bound="ChartStyle")


#
# controller: class ChartDrawer
#


class ChartDrawer(Generic[T_Model, T_Style], ABC):
    """
    Base class for chart drawers.

    Implementations must define a :meth:`~ChartDrawer._draw` method.

    :param title: title of the chart
    :param model: the model containing the underlying data represented by the chart
    :param style: the style of the chart
    """

    def __init__(self, model: T_Model, style: T_Style, title: str) -> None:
        self._model = model
        self._style = style
        self._title = title

    @property
    def title(self) -> str:
        return self._title

    @property
    def model(self) -> T_Model:
        return self._model

    @property
    def style(self) -> T_Style:
        return self._style

    def draw(self) -> None:
        """Draw the chart."""
        self.style.drawing_start(self._title)
        self._draw()
        self.style.drawing_finalize()

    @abstractmethod
    def _draw(self) -> None:
        pass


#
# view: class ChartStyle
#


class ChartStyle(ABC):
    """
    Base class for a drawer style.

    Implementations must define :meth:`~ChartStyle.draw_title`.
    """

    @abstractmethod
    def drawing_start(self, title: str) -> None:
        """
        Start drawing a new chart.
        :title: the chart title
        """
        pass

    def drawing_finalize(self) -> None:
        """
        Finalize the chart.

        Does nothing by default, can optionally be overloaded
        """
        pass


class MatplotStyle(ChartStyle, ABC):
    """Matplotlib drawer style.

    Implementations must define :meth:`~ChartStyle.draw_title`.
    :param ax: optional axes object to draw on; if ``Null`` use pyplot's current axes
    """

    def __init__(self, ax: Optional[Axes] = None) -> None:
        super().__init__()
        self._ax = ax = plt.gca() if ax is None else ax
        self._renderer = ax.figure.canvas.get_renderer()

    @property
    def ax(self) -> Axes:
        """
        The matplot :class:`~matplotlib.axes.Axes` object to draw the chart in.
        """
        return self._ax

    def drawing_start(self, title: str) -> None:
        """Draw the title of the chart."""
        self.ax.set_title(label=title)

    def text_size(
        self, text: str, x: Optional[float] = None, y: Optional[float] = None, **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate the horizontal and vertical size of the given text in axis units.
        Constructs a :class:`matplotlib.text.Text` artist then calculates it size
        relative to the axis managed by this style object (attribute `ax`)
        For non-linear axis scales text size differs depending on placement,
        so the intended placement (in data coordinates) should be provided

        :param text: text to calculate the size for
        :param x: intended horizontal text placement (optional, defaults to left of
            view)
        :param y: intended vertical text placement (optional, defaults to bottom of
            view)
        :param kwargs: additional arguments to use when constructing the
            :class:`~matplotlib.text.Text` artist, e.g., rotation
        :return: tuple `(width, height)` in absolute axis units
        """

        ax = self.ax

        if x is None or y is None:
            x0, y0, _, _ = ax.dataLim.bounds
            if x is None:
                x = x0
            if y is None:
                y = y0

        fig = ax.figure

        extent = mt.Text(x, y, text, figure=fig, **kwargs).get_window_extent(
            fig.canvas.get_renderer()
        )

        (x0, y0), (x1, y1) = ax.transData.inverted().transform(extent)

        return abs(x1 - x0), abs(y1 - y0)


class TextStyle(ChartStyle, ABC):
    """
    Plain text drawing style.

    :param width: the maximum width available to render the text, defaults to 80
    :param out: the output stream this style instance writes to, or `stdout` if \
      `None` is passed (defaults to `None`)
    """

    def __init__(self, out: TextIO = None, width: int = 80) -> None:
        if width <= 0:
            raise ValueError(
                f"arg width expected to be positive integer but is {width}"
            )
        self._out = sys.stdout if out is None else out
        self._width = width

    @property
    def out(self) -> TextIO:
        """
        The output stream this style instance writes to.
        """
        return self._out

    @property
    def width(self) -> int:
        """
        The maximum width of the text to be produced.
        """
        return self._width


#
# Rgba color class for use in  MatplotStyles
#

RgbaColor = Tuple[float, float, float, float]
