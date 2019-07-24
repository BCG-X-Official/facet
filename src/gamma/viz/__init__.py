"""
MVC-based classes for drawing charts.
"""
from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

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
        self._draw()
        self.style.draw_title(self._title)

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
    def draw_title(self, title: str) -> None:
        """
        Draw the diagram title.
        :title: the diagram title
        """
        pass


class MatplotStyle(ChartStyle, ABC):
    """Matplotlib drawer style.

    Implementations must define :meth:`~ChartStyle.draw_title`.
    :param ax: drawn axes
    """

    def __init__(self, ax: Optional[Axes] = None) -> None:
        super().__init__()
        self._ax = ax

    @property
    def ax(self) -> Axes:
        return self._ax

    @ax.setter
    def ax(self, new_ax: Axes):
        self._ax = new_ax


#
# Rgba color class for use in  MatplotStyles
#

RgbaColor = Tuple[float, float, float, float]
