"""
MVC-based classes for drawing charts.
"""
import sys
from abc import ABC, abstractmethod
from typing import *
from typing import TextIO

import matplotlib.pyplot as plt
from matplotlib import text as mt
from matplotlib.axes import Axes

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

    def __init__(self, title: str, model: T_Model, style: T_Style) -> None:
        self._title = title
        self._model = model
        self._style = style

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

    def text_extent(self, text: str, **kwargs) -> Tuple[float, float]:
        """
        Calculate the horizontal and vertical extend of the given text in axis units.
        Constructs a :class:`matplotlib.text.Text` artist then calculates it size
        relative to the axis managed by this style object (attribute `ax`)

        :param text: text to calculate the size for
        :param kwargs: additional arguments to use when constructing the
          :class:`~matplotlib.text.Text` artist, e.g., rotation
        :return: tuple `(width, height)` in axis units
        """
        (x0, y0), (x1, y1) = self.ax.transData.inverted().transform(
            mt.Text(0, 0, text, figure=self.ax.figure, **kwargs).get_window_extent(
                self._renderer
            )
        )
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
