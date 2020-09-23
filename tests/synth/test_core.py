"""
Test `sinethesizer.synth.core` module.

Author: Nikolay Lysenko
"""


import functools
import math
from typing import Dict

import numpy as np
import pytest

from sinethesizer.effects.stereo import apply_haas_effect
from sinethesizer.envelopes.misc import create_constant_envelope
from sinethesizer.synth.core import (
    Event, Instrument, ModulatedWave, Modulator, Partial,
    adjust_envelope_duration, generate_modulated_wave, generate_partial,
    sum_two_sounds, synthesize
)
from sinethesizer.synth.event_to_amplitude_factor import (
    compute_amplitude_factor_as_power_of_velocity
)


@pytest.mark.parametrize(
    "envelope, required_len, expected",
    [
        (np.array([1, 2, 3]), 2, np.array([1, 2])),
        (np.array([1, 2, 3]), 3, np.array([1, 2, 3])),
        (np.array([1, 2, 3]), 5, np.array([1, 2, 3, 3, 3])),
    ]
)
def test_adjust_envelope_duration(
        envelope: np.ndarray, required_len: int, expected: np.ndarray
) -> None:
    """Test `adjust_envelope_duration` function."""
    result = adjust_envelope_duration(envelope, required_len)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "wave, frequency, event, expected",
    [
        (
            # `wave`
            ModulatedWave(
                waveform='sine',
                phase=math.pi / 2,
                amplitude_envelope_fn=functools.partial(
                    create_constant_envelope,
                    value=1
                ),
                modulator=None
            ),
            # `frequency`
            2,
            # `event`
            Event(
                instrument='any_instrument',
                start_time=1.0,
                duration=1.0,
                frequency=1.0,
                velocity=1.0,
                effects='',
                frame_rate=20
            ),
            # `expected`
            np.array([
                [
                    1.0, 0.80901699, 0.30901699, -0.30901699, -0.80901699,
                    -1.0, -0.80901699, -0.30901699, 0.30901699, 0.80901699,
                    1.0, 0.80901699, 0.30901699, -0.30901699, -0.80901699,
                    -1.0, -0.80901699, -0.30901699, 0.30901699, 0.80901699,
                ],
                [
                    1.0, 0.80901699, 0.30901699, -0.30901699, -0.80901699,
                    -1.0, -0.80901699, -0.30901699, 0.30901699, 0.80901699,
                    1.0, 0.80901699, 0.30901699, -0.30901699, -0.80901699,
                    -1.0, -0.80901699, -0.30901699, 0.30901699, 0.80901699,
                ],
            ])
        ),
        (
            # `wave`
            ModulatedWave(
                waveform='sine',
                phase=math.pi / 2,
                amplitude_envelope_fn=functools.partial(
                    create_constant_envelope,
                    value=1
                ),
                modulator=Modulator(
                    waveform='sine',
                    frequency_ratio_numerator=3,
                    frequency_ratio_denominator=2,
                    phase=0,
                    modulation_index_envelope_fn=functools.partial(
                        create_constant_envelope,
                        value=3
                    )
                )
            ),
            # `frequency`
            2,
            # `event`
            Event(
                instrument='any_instrument',
                start_time=1.0,
                duration=1.0,
                frequency=1.0,
                velocity=1.0,
                effects='',
                frame_rate=20
            ),
            # `expected`
            np.array([
                [
                    1.0, -0.56677191, 0.73174451, -0.42209869, -0.02573332,
                    1.0, -0.02573332, -0.42209869, 0.73174451, -0.56677191,
                    1.0, -0.56677191, 0.73174451, -0.42209869, -0.02573332,
                    1.0, -0.02573332, -0.42209869, 0.73174451, -0.56677191,
                ],
                [
                    1.0, -0.56677191, 0.73174451, -0.42209869, -0.02573332,
                    1.0, -0.02573332, -0.42209869, 0.73174451, -0.56677191,
                    1.0, -0.56677191, 0.73174451, -0.42209869, -0.02573332,
                    1.0, -0.02573332, -0.42209869, 0.73174451, -0.56677191,
                ],
            ])
        ),
    ]
)
def test_generate_modulated_wave(
        wave: ModulatedWave, frequency: float, event: Event,
        expected: np.ndarray
) -> None:
    """Test `generate_modulated_wave` function."""
    result = generate_modulated_wave(wave, frequency, event)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "partial, event, expected",
    [
        (
            # `partial`
            Partial(
                wave=ModulatedWave(
                    waveform='sine',
                    phase=0,
                    amplitude_envelope_fn=functools.partial(
                        create_constant_envelope,
                        value=1
                    ),
                    modulator=None
                ),
                frequency_ratio=2.0,
                amplitude_ratio=0.5,
                event_to_amplitude_factor_fn=functools.partial(
                    compute_amplitude_factor_as_power_of_velocity,
                    power=1
                ),
                detuning_to_amplitude={0.0: 1.0},
                random_detuning_range=0.0,
                effects=[
                    functools.partial(
                        apply_haas_effect,
                        location=-1, max_channel_delay=0.1
                    )
                ]
            ),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0.0,
                duration=1.0,
                frequency=1.0,
                velocity=1.0,
                effects='',
                frame_rate=20
            ),
            # `expected`
            np.array([
                [
                    0.0, 0.293892626, 0.475528258, 0.475528258, 0.293892626,
                    0.0, -0.293892626, -0.475528258, -0.475528258, -0.293892626,
                    0.0, 0.293892626, 0.475528258, 0.475528258, 0.293892626,
                    0.0, -0.293892626, -0.475528258, -0.475528258, -0.293892626,
                    0.0, 0.0,
                ],
                [
                    0.0, 0.0,
                    0.0, 0.293892626, 0.475528258, 0.475528258, 0.293892626,
                    0.0, -0.293892626, -0.475528258, -0.475528258, -0.293892626,
                    0.0, 0.293892626, 0.475528258, 0.475528258, 0.293892626,
                    0.0, -0.293892626, -0.475528258, -0.475528258, -0.293892626,
                ]
            ])
        ),
    ]
)
def test_generate_partial(
        partial: Partial, event: Event, expected: np.ndarray
) -> None:
    """Test `generate_partial` function."""
    result = generate_partial(partial, event)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "first_sound, second_sound, expected",
    [
        (
            np.array([[1, 2, 3], [2, 3, 4]]),
            np.array([[7, 9], [8, 9]]),
            np.array([[8, 11, 3], [10, 12, 4]])
        ),
        (
            np.array([[7, 9], [8, 9]]),
            np.array([[1, 2, 3], [2, 3, 4]]),
            np.array([[8, 11, 3], [10, 12, 4]])
        ),
    ]
)
def test_sum_two_sounds(
        first_sound: np.ndarray, second_sound: np.ndarray, expected: np.ndarray
) -> None:
    """Test `sum_two_sounds` function."""
    result = sum_two_sounds(first_sound, second_sound)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "event, instruments_registry, expected",
    [
        (
            # `event`
            Event(
                instrument='sine',
                start_time=1.0,
                duration=0.5,
                frequency=1.0,
                velocity=1.0,
                effects='',
                frame_rate=20,
            ),
            # `instruments_registry`
            {
                'sine': Instrument(
                    partials=[
                        Partial(
                            wave=ModulatedWave(
                                waveform='sine',
                                phase=0,
                                amplitude_envelope_fn=functools.partial(
                                    create_constant_envelope,
                                    value=1
                                ),
                                modulator=None
                            ),
                            frequency_ratio=1.0,
                            amplitude_ratio=1.0,
                            event_to_amplitude_factor_fn=functools.partial(
                                compute_amplitude_factor_as_power_of_velocity,
                                power=1
                            ),
                            detuning_to_amplitude={0.0: 1.0},
                            random_detuning_range=0.0,
                            effects=[]
                        )
                    ],
                    amplitude_scaling=1.0,
                    effects=[]
                )
            },
            # `expected`
            np.array([
                [
                    0.0, 0.309017, 0.5877853, 0.809017, 0.9510565,
                    1.0, 0.9510565, 0.809017, 0.5877853, 0.309017
                ],
                [
                    0.0, 0.309017, 0.5877853, 0.809017, 0.9510565,
                    1.0, 0.9510565, 0.809017, 0.5877853, 0.309017
                ],
            ])
        ),
        (
            # `event`
            Event(
                instrument='sine',
                start_time=1.0,
                duration=0.5,
                frequency=1.0,
                velocity=1.0,
                effects='[{"name": "haas", "location": -1, "max_channel_delay": 0.05}]',
                frame_rate=20,
            ),
            # `instruments_registry`
            {
                'sine': Instrument(
                    partials=[
                        Partial(
                            wave=ModulatedWave(
                                waveform='sine',
                                phase=0,
                                amplitude_envelope_fn=functools.partial(
                                    create_constant_envelope,
                                    value=1
                                ),
                                modulator=None
                            ),
                            frequency_ratio=1.0,
                            amplitude_ratio=1.0,
                            event_to_amplitude_factor_fn=functools.partial(
                                compute_amplitude_factor_as_power_of_velocity,
                                power=1
                            ),
                            detuning_to_amplitude={0.0: 1.0},
                            random_detuning_range=0.0,
                            effects=[]
                        )
                    ],
                    amplitude_scaling=1.0,
                    effects=[]
                )
            },
            # `expected`
            np.array([
                [
                    0.0, 0.309017, 0.5877853, 0.809017, 0.9510565,
                    1.0, 0.9510565, 0.809017, 0.5877853, 0.309017, 0.0
                ],
                [
                    0.0, 0.0, 0.309017, 0.5877853, 0.809017, 0.9510565,
                    1.0, 0.9510565, 0.809017, 0.5877853, 0.309017
                ],
            ])
        ),
    ]
)
def test_synthesize(
        event: Event, instruments_registry: Dict[str, Instrument],
        expected: np.ndarray
) -> None:
    """Test `synthesize` function."""
    result = synthesize(event, instruments_registry)
    np.testing.assert_almost_equal(result, expected)
