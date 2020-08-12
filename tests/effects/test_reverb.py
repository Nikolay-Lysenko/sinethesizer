"""
Test `sinethesizer.effects.reverb` module.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, Optional

import numpy as np
import pytest

from sinethesizer.effects.reverb import apply_reverb


@pytest.mark.parametrize(
    "sound, sound_info, first_reflection_delay, decay_duration, "
    "amplitude_random_range, n_early_reflections, early_reflections_delay, "
    "diffusion_delay_factor, diffusion_delay_random_range, "
    "original_sound_gain, reverberations_gain, keep_peak_volume, random_seed, "
    "expected",
    [
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 5, 10, 5, 1],
                [1, 5, 10, 5, 4, 3, 2, 1]
            ]),
            # `sound_info`
            {'frame_rate': 4},
            # `first_reflection_delay`
            2.0,
            # `decay_duration`
            1.8,
            # `amplitude_random_range`
            0.0,
            # `n_early_reflections`
            2,
            # `early_reflections_delay`
            1.0,
            # `diffusion_delay_factor`
            0.5,
            # `diffusion_delay_random_range`
            0.0,
            # `original_sound_gain`
            1.0,
            # `reverberations_gain`
            1.0,
            # `keep_peak_volume`
            False,
            # `random_seed`
            0,
            # `expected`
            np.array([
                [
                    1, 2, 3, 4, 5, 10, 5, 1,
                    1, 2, 3, 4, 5.44444444, 10.88888889, 6.5, 3.13888889,
                    2.77777778, 5.19444444, 3.16666667, 2.25, 1.11111111,
                    0.30555556, 0.02777778
                ],
                [
                    1, 5, 10, 5, 4, 3, 2, 1,
                    1, 5, 10, 5, 4.44444444, 5.22222222, 6.61111111,
                    4.08333333, 3.58333333, 2.44444444, 1.69444444,
                    1.05555556, 0.41666667, 0.22222222, 0.02777778
                ]
            ])
        ),
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 5, 10, 5, 1],
                [1, 5, 10, 5, 4, 3, 2, 1]
            ]),
            # `sound_info`
            {'frame_rate': 4},
            # `first_reflection_delay`
            2.0,
            # `decay_duration`
            1.8,
            # `amplitude_random_range`
            0.0,
            # `n_early_reflections`
            2,
            # `early_reflections_delay`
            1.0,
            # `diffusion_delay_factor`
            0.5,
            # `diffusion_delay_random_range`
            0.0,
            # `original_sound_gain`
            1.0,
            # `reverberations_gain`
            1.0,
            # `keep_peak_volume`
            True,
            # `random_seed`
            0,
            # `expected`
            np.array([
                [
                    0.91836735, 1.83673469, 2.75510204, 3.67346939, 4.59183673,
                    9.18367347, 4.59183673, 0.91836735, 0.91836735, 1.83673469,
                    2.75510204, 3.67346939, 5, 10, 5.96938776, 2.88265306,
                    2.55102041, 4.77040816, 2.90816327, 2.06632653, 1.02040816,
                    0.28061224, 0.0255102
                ],
                [
                    0.91836735, 4.59183673, 9.18367347, 4.59183673, 3.67346939,
                    2.75510204, 1.83673469, 0.91836735, 0.91836735, 4.59183673,
                    9.18367347, 4.59183673, 4.08163265, 4.79591837, 6.07142857,
                    3.75, 3.29081633, 2.24489796, 1.55612245, 0.96938776,
                    0.38265306, 0.20408163, 0.0255102
                ]
            ])
        ),
    ]
)
def test_apply_reverb(
        sound: np.ndarray, sound_info: Dict[str, Any],
        first_reflection_delay: float,
        decay_duration: float,
        amplitude_random_range: float,
        n_early_reflections: int,
        early_reflections_delay: float,
        diffusion_delay_factor: float,
        diffusion_delay_random_range: float,
        original_sound_gain: float,
        reverberations_gain: float,
        keep_peak_volume: bool,
        random_seed: Optional[int],
        expected: np.ndarray
) -> None:
    """Test `apply_reverb` function."""
    result = apply_reverb(
        sound, sound_info, first_reflection_delay, decay_duration,
        amplitude_random_range, n_early_reflections, early_reflections_delay,
        diffusion_delay_factor, diffusion_delay_random_range,
        original_sound_gain, reverberations_gain, keep_peak_volume, random_seed
    )
    np.testing.assert_almost_equal(result, expected)
