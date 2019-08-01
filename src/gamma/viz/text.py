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
Utilities for text rendering
"""
import logging
from typing import *

from gamma import ListLike, MatrixLike

TextCoordinates = Tuple[Union[int, slice], Union[int, slice]]

log = logging.getLogger(__name__)


class CharacterMatrix:
    """
    A matrix of characters, indexed by rows and columns.

    :param n_rows: the matrix height
    :param n_columns: the matrix width
    """

    def __init__(self, n_rows: int, n_columns: int):
        if n_columns <= 0:
            raise ValueError(f"arg width must be positive but is {n_columns}")
        if n_rows <= 0:
            raise ValueError(f"arg height must be positive but is {n_rows}")
        self._n_columns = n_columns
        self._matrix = [[" " for _ in range(n_columns)] for _ in range(n_rows)]

    @property
    def n_rows(self) -> int:
        """
        The height of this matrix.

        Same as ``len(self)``.
        """
        return len(self._matrix)

    @property
    def n_columns(self) -> int:
        """
        The height of this matrix.
        """
        return self._n_columns

    def lines(self) -> Iterable[str]:
        """
        :return: the lines in this matrix as strings
        """
        return ("".join(line) for line in self._matrix)

    @staticmethod
    def _key_as_slices(key: TextCoordinates) -> Tuple[slice, slice]:
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
        return self.n_rows

    def __getitem__(self, key: TextCoordinates):
        rows, columns = self._key_as_slices(key)
        return "\n".join("".join(line[columns]) for line in self._matrix[rows])

    def __setitem__(self, key: TextCoordinates, value: Any) -> None:
        rows, columns = self._key_as_slices(key)
        value = str(value)
        single_char = len(value) == 1
        positions = range(*columns.indices(self.n_columns))
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
