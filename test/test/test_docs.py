"""
Validate docstrings and type hints.
"""

from pytools.api import DocValidator


def test_doc() -> None:
    assert DocValidator(
        root_dir="src"
    ).validate_doc(), "docstrings and type hints are valid"
