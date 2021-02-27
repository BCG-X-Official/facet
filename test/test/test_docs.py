"""
Test docstrings.
"""

from pytools.api.doc import DocValidator


def test_docstrings() -> None:
    assert DocValidator(root_dir="src").validate_docstrings(), "docstrings are valid"
