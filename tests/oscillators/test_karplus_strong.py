"""
Test `sinethesizer.oscillators.karplus_strong` module.

Author: Nikolay Lysenko
"""


import pytest

from sinethesizer.oscillators.karplus_strong import (
    generate_karplus_strong_wave
)


@pytest.mark.parametrize(
    "frequency, duration_in_frames, frame_rate",
    [
        (50, 300, 300),
    ]
)
def test_generate_karplus_strong_wave(
        frequency: float, duration_in_frames: int, frame_rate: int
) -> None:
    """Test `generate_karplus_strong_wave` function."""
    result = generate_karplus_strong_wave(
        frequency, duration_in_frames, frame_rate
    )
    assert len(result) == duration_in_frames
