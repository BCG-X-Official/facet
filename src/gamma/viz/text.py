"""
Utilities for text rendering
"""
import logging
from typing import *

from gamma import ListLike, MatrixLike

TextCoordinates = Tuple[Union[int, slice], Union[int, slice]]

log = logging.getLogger(__name__)


class CharacterMatrix:
    def __init__(self, height: int, width: int):
        if width <= 0:
            raise ValueError(f"arg width must be positive but is {width}")
        if height <= 0:
            raise ValueError(f"arg height must be positive but is {height}")
        self._width = width
        self._matrix = [[" " for _ in range(width)] for _ in range(height)]

    @property
    def height(self) -> int:
        """
        The height of this matrix.

        Same as ``len(self)``.
        """
        return len(self._matrix)

    @property
    def width(self) -> int:
        """
        The height of this matrix.
        """
        return self._width

    def lines(self) -> Iterable[str]:
        return ("".join(line) for line in self._matrix)

    def _key_as_slices(self, key: TextCoordinates) -> Tuple[slice, slice]:
        def _to_slice(index: Union[int, slice]) -> slice:
            if isinstance(index, int):
                return slice(index, index + 1)
            else:
                return index

        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"expected (row, column) tuple but got {key}")

        row, column = key
        return _to_slice(row), _to_slice(column)

    def __str__(self) -> str:
        return "\n".join(self.lines())

    def __len__(self) -> int:
        return self.height

    def __getitem__(self, key: TextCoordinates):
        rows, columns = self._key_as_slices(key)
        return "\n".join("".join(line[columns]) for line in self._matrix[rows])

    def __setitem__(self, key: TextCoordinates, value: Any) -> None:
        rows, columns = self._key_as_slices(key)
        value = str(value)
        single_char = len(value) == 1
        positions = range(*columns.indices(self.width))
        for line in self._matrix[rows]:
            if single_char:
                for pos in positions:
                    line[pos] = value
            else:
                for pos, char in zip(positions, value):
                    line[pos] = char


def format_table(
    headings: ListLike[str],
    data: MatrixLike,
    formats: Optional[ListLike[Optional[str]]] = None,
) -> str:
    """
    Print a formatted text table
    :param headings: the table headings
    :param data: the table data, as a 2D list-like organised as a list of rows
    :param formats: formatting strings for data in each row (optional); \
        uses `str()` conversion for any formatting strings stated as `None`
    :return: the formatted table as a multi-line string
    """
    n_columns = len(headings)

    if formats is None:
        formats = [None] * n_columns
    elif len(formats) != n_columns:
        raise ValueError("arg formats must have the same length as arg headings")

    def _formatted(item: Any, format_string: str) -> str:
        if format_string is None:
            return str(item)
        else:
            return f"{item:{format_string}}"

    def _make_row(items: ListLike):
        if len(items) != n_columns:
            raise ValueError(
                "rows in data matrix must have the same length as arg headings"
            )
        return [
            _formatted(item, format_string)
            for item, format_string in zip(items, formats)
        ]

    rows = [_make_row(items) for items in data]

    column_widths = [
        max(column_lengths)
        for column_lengths in zip(
            *((len(item) for item in row) for row in (headings, *rows))
        )
    ]

    dividers = ["-" * column_width for column_width in column_widths]

    return "\n".join(
        (
            *(
                " ".join(
                    (
                        f"{item:{column_width}s}"
                        for item, column_width in zip(row, column_widths)
                    )
                )
                for row in (headings, dividers, *rows)
            ),
            "",
        )
    )
