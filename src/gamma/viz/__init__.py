"""
MVC-based classes for drawing charts.
"""
from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

from matplotlib.axes import Axes

T_Model = TypeVar("T_Model")
T_Style = TypeVar("T_Style", bound="ChartStyle")


#
# controller: class ChartDrawer
#


class ChartDrawer(Generic[T_Model, T_Style], ABC):
    """Chart drawer.

    :param title: title of the chart
    :param model: the data model of the chart
    :param style: the chart style of the chart
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
        self.style.draw_title(self._title)
        self._draw()

    @abstractmethod
    def _draw(self) -> None:
        pass


#
# view: class ChartStyle
#


class ChartStyle(ABC):
    """
    Chart style.

    Implementations must define :meth:`ChartStyle.draw_title`.
    """

    @abstractmethod
    def draw_title(self, title: str) -> None:
        """
        Draw the diagram title.
        :title: the diagram title
        """
        pass


class MatplotStyle(ChartStyle, ABC):
    """

    """

    def __init__(self, ax: Axes) -> None:
        super().__init__()
        self._ax = ax

    @property
    def ax(self) -> Axes:
        return self._ax


#
# Rgba color class for use in  MatplotStyles
#

RgbaColor = Tuple[float, float, float, float]
