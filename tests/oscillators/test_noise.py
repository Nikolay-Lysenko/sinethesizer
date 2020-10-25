"""
Test `sinethesizer.oscillators.noise` module.

Author: Nikolay Lysenko
"""


import pytest
from scipy.signal import spectrogram

from sinethesizer.oscillators.noise import generate_power_law_noise


@pytest.mark.parametrize(
    "duration_in_frames, frame_rate, psd_decay_order, exponential_step, "
    "nperseg",
    [
        (1000, 10000, 0, 2, 100),
        (1000, 10000, 1, 2, 100),
        (1000, 10000, 2, 2, 100),
    ]
)
def test_generate_power_law_noise(
        duration_in_frames: int, frame_rate: int, psd_decay_order: float,
        exponential_step: float, nperseg: int
) -> None:
    """Test `generate_power_law_noise` function."""
    noise = generate_power_law_noise(
        duration_in_frames, frame_rate, psd_decay_order, exponential_step
    )
    spc = spectrogram(noise, frame_rate, nperseg=nperseg)[2]
    result = spc.sum(axis=1)
    if psd_decay_order > 0:
        assert result[0] > result[-1]
        assert result[10] > result[30]
