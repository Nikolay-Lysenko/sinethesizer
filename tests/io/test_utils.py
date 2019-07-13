"""
Test `sinethesizer.io.utils` module.

Author: Nikolay Lysenko
"""


import pytest

from sinethesizer.io.utils import (
    get_list_of_notes, convert_note_to_frequency
)


def test_get_list_of_notes() -> None:
    """Test `get_list_of_notes` function."""
    result = get_list_of_notes()
    assert len(result) == 88
    assert result[0] == 'A0'
    assert result[-1] == 'C8'


@pytest.mark.parametrize(
    "note, expected",
    [
        ('A4', 440.0),
        ('Cs5', 554.37),
        ('E3', 164.81),
        ('Gs1', 51.91),
        ('C8', 4186.01)
    ]
)
def test_convert_note_to_frequency(
        note: str, expected: float
) -> None:
    """Test `convert_note_to_frequency` function."""
    result = convert_note_to_frequency(note)
    assert round(result, 2) == expected
