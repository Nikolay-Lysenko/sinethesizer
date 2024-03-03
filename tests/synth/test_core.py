"""
Test `sinethesizer.synth.core` module.

Author: Nikolay Lysenko
"""


import functools
import math

import numpy as np
import pytest

from sinethesizer.effects.stereo import apply_stereo_delay
from sinethesizer.envelopes.misc import create_constant_envelope
from sinethesizer.synth.core import (
    Event, Instrument, ModulatedWave, Modulator, Partial,
    adjust_envelope_duration, generate_modulated_wave, generate_partial,
    introduce_quasiperiodicity, synthesize
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
    "phase_modulator, n_frames, frame_rate, frequency,"
    "quasiperiodic_bandwidth, quasiperiodic_breakpoints_frequency,"
    "upper_threshold",
    [
        (
            None, 30000, 10000, 100, 1, 10, 2 ** (0.5 / 12) - 1
        ),
        (
            # In this test, `phase_modulator` can contain only zeros.
            np.zeros(30000), 30000, 10000, 100, 1, 10, 2 ** (0.5 / 12) - 1
        ),
    ]
)
def test_introduce_quasiperiodicity(
        phase_modulator: np.ndarray, n_frames: int, frame_rate: int,
        frequency: float, quasiperiodic_bandwidth: float,
        quasiperiodic_breakpoints_frequency: float, upper_threshold: float
) -> None:
    """Test `introduce_quasiperiodicity` function."""
    modulator = introduce_quasiperiodicity(
        phase_modulator, n_frames, frame_rate, frequency,
        quasiperiodic_bandwidth, quasiperiodic_breakpoints_frequency
    )
    assert len(modulator) == n_frames
    derivative_of_modulator = np.diff(modulator) * frame_rate
    scaled_derivative = derivative_of_modulator / frequency / (2 * np.pi)
    scaled_derivative = np.abs(scaled_derivative)
    max_relative_deviation = np.max(scaled_derivative)
    assert max_relative_deviation <= upper_threshold


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
                amplitude_modulator=None,
                phase_modulator=None,
                quasiperiodic_bandwidth=0,
                quasiperiodic_breakpoints_frequency=10
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
                amplitude_envelope_fn=functools.partial(
                    create_constant_envelope,
                    value=1
                ),
                phase=math.pi / 2,
                amplitude_modulator=None,
                phase_modulator=Modulator(
                    waveform='sine',
                    carrier_frequency_ratio=2,
                    modulator_frequency_ratio=3,
                    modulation_index_envelope_fn=functools.partial(
                        create_constant_envelope,
                        value=3
                    ),
                    phase=0,
                    use_ring_modulation=False
                ),
                quasiperiodic_bandwidth=0,
                quasiperiodic_breakpoints_frequency=10
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
                    amplitude_modulator=None,
                    phase_modulator=None,
                    quasiperiodic_bandwidth=0,
                    quasiperiodic_breakpoints_frequency=10
                ),
                frequency_ratio=2.0,
                amplitude_ratio=0.5,
                event_to_amplitude_factor_fn=functools.partial(
                    compute_amplitude_factor_as_power_of_velocity,
                    power=1
                ),
                detuning_to_amplitude={0.0: 1.0},
                random_detuning_range=0.0,
                effects=[functools.partial(apply_stereo_delay, delay=-0.1)]
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
                    amplitude_modulator=None,
                    phase_modulator=None,
                    quasiperiodic_bandwidth=0,
                    quasiperiodic_breakpoints_frequency=10
                ),
                frequency_ratio=2.0,
                amplitude_ratio=0.5,
                event_to_amplitude_factor_fn=functools.partial(
                    compute_amplitude_factor_as_power_of_velocity,
                    power=1
                ),
                detuning_to_amplitude={0.0: 1.0},
                random_detuning_range=0.0,
                effects=[functools.partial(apply_stereo_delay, delay=-0.1)]
            ),
            # `event`
            Event(
                instrument='any_instrument',
                start_time=0.0,
                duration=1.0,
                frequency=15.0,
                velocity=1.0,
                effects='',
                frame_rate=20
            ),
            # `expected`
            np.array([[], []])
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
                                amplitude_envelope_fn=functools.partial(
                                    create_constant_envelope,
                                    value=1
                                ),
                                phase=0,
                                amplitude_modulator=None,
                                phase_modulator=None,
                                quasiperiodic_bandwidth=0,
                                quasiperiodic_breakpoints_frequency=10
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
                effects='[{"name": "stereo_delay", "delay": -0.05}]',
                frame_rate=20,
            ),
            # `instruments_registry`
            {
                'sine': Instrument(
                    partials=[
                        Partial(
                            wave=ModulatedWave(
                                waveform='sine',
                                amplitude_envelope_fn=functools.partial(
                                    create_constant_envelope,
                                    value=1
                                ),
                                phase=0,
                                amplitude_modulator=None,
                                phase_modulator=None,
                                quasiperiodic_bandwidth=0,
                                quasiperiodic_breakpoints_frequency=10
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
        event: Event, instruments_registry: dict[str, Instrument], expected: np.ndarray
) -> None:
    """Test `synthesize` function."""
    result = synthesize(event, instruments_registry)
    np.testing.assert_almost_equal(result, expected)
