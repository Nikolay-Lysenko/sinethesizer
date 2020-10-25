"""
Test `sinethesizer.effects.reverb` module.

Author: Nikolay Lysenko
"""


from typing import Optional

import numpy as np
import pytest

from sinethesizer.effects.reverb import apply_reverb
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "sound, event, first_reflection_delay, decay_duration, "
    "amplitude_random_range, n_early_reflections, early_reflections_delay, "
    "diffusion_delay_factor, diffusion_delay_random_range, "
    "late_reflections_decay_power, original_sound_gain, reverberations_gain, "
    "random_seed, keep_peak_volume, expected",
    [
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 5, 10, 5, 1],
                [1, 5, 10, 5, 4, 3, 2, 1]
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=4
            ),
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
            # `late_reflections_decay_power`
            10.0,
            # `original_sound_gain`
            1.0,
            # `reverberations_gain`
            1.0,
            # `random_seed`
            0,
            # `keep_peak_volume`
            False,
            # `expected`
            np.array([
                [
                    1, 2, 3, 4, 5, 10, 5, 1,
                    0.00517892437, 0.0103578487, 0.0155367731, 0.0207156975,
                    0.0262673218, 0.0525346436, 0.0271127032, 0.00692147148,
                    0.00226701291, 0.00428227858, 0.00257054458, 0.00163143618,
                    0.00101775133, 0.000358903552, 0.000051784429
                ],
                [
                    1, 5, 10, 5, 4, 3, 2, 1,
                    0.00517892437, 0.0258946219, 0.0517892437, 0.0258946219,
                    0.0210883974, 0.0174002729, 0.0141848298, 0.00759411567,
                    0.00274953608, 0.00213585123, 0.00140424771, 0.0008797819,
                    0.000355316102, 0.000203550265, 0.000051784429
                ]
            ])
        ),
        (
            # `sound`
            np.array([
                [1, 2, 3, 4, 5, 10, 5, 1],
                [1, 5, 10, 5, 4, 3, 2, 1]
            ]),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0,
                duration=1,
                frequency=440,
                velocity=1,
                effects='',
                frame_rate=4
            ),
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
            # `late_reflections_decay_power`
            10.0,
            # `original_sound_gain`
            1.0,
            # `reverberations_gain`
            1.0,
            # `random_seed`
            0,
            # `keep_peak_volume`
            True,
            # `expected`
            np.array([
                [
                    1, 2, 3, 4, 5, 10, 5, 1,
                    0.00517892437, 0.0103578487, 0.0155367731, 0.0207156975,
                    0.0262673218, 0.0525346436, 0.0271127032, 0.00692147148,
                    0.00226701291, 0.00428227858, 0.00257054458, 0.00163143618,
                    0.00101775133, 0.000358903552, 0.000051784429
                ],
                [
                    1, 5, 10, 5, 4, 3, 2, 1,
                    0.00517892437, 0.0258946219, 0.0517892437, 0.0258946219,
                    0.0210883974, 0.0174002729, 0.0141848298, 0.00759411567,
                    0.00274953608, 0.00213585123, 0.00140424771, 0.0008797819,
                    0.000355316102, 0.000203550265, 0.000051784429
                ]
            ])
        ),
    ]
)
def test_apply_reverb(
        sound: np.ndarray, event: Event,
        first_reflection_delay: float,
        decay_duration: float,
        amplitude_random_range: float,
        n_early_reflections: int,
        early_reflections_delay: float,
        diffusion_delay_factor: float,
        diffusion_delay_random_range: float,
        late_reflections_decay_power: float,
        original_sound_gain: float,
        reverberations_gain: float,
        keep_peak_volume: bool,
        random_seed: Optional[int],
        expected: np.ndarray
) -> None:
    """Test `apply_reverb` function."""
    result = apply_reverb(
        sound, event, first_reflection_delay, decay_duration,
        amplitude_random_range, n_early_reflections, early_reflections_delay,
        diffusion_delay_factor, diffusion_delay_random_range,
        late_reflections_decay_power, original_sound_gain, reverberations_gain,
        random_seed, keep_peak_volume
    )
    print(result)
    np.testing.assert_almost_equal(result, expected)
