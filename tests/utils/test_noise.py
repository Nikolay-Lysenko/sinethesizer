"""
Test `sinethesizer.utils.noise` module.

Author: Nikolay Lysenko
"""


import numpy as np
import pytest
from scipy.signal import spectrogram

from sinethesizer.utils.noise import generate_power_law_noise


@pytest.mark.parametrize(
    "xs, frame_rate, psd_decay_order, n_equalizer_points, nperseg",
    [
        (np.linspace(0, 1000, 1000), 10000, 0, 300, 100),
        (np.linspace(0, 1000, 1000), 10000, 1, 300, 100),
        (np.linspace(0, 1000, 1000), 10000, 2, 300, 100),
    ]
)
def test_generate_power_law_noise(
        xs: np.ndarray, frame_rate: int, psd_decay_order: float,
        n_equalizer_points: int, nperseg: int
) -> None:
    """Test `generate_power_law_noise` function."""
    noise = generate_power_law_noise(
        xs, frame_rate, psd_decay_order, n_equalizer_points
    )
    spc = spectrogram(noise, frame_rate, nperseg=nperseg)[2]
    result = spc.sum(axis=1)
    if psd_decay_order > 0:
        assert result[0] > result[-1]
        assert result[10] > result[30]
