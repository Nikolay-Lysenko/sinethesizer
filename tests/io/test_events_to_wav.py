"""
Test `sinethesizer.io.events_to_wav` module.

Author: Nikolay Lysenko
"""


import functools
from typing import Any, Dict, List

import pytest
import numpy as np

from sinethesizer.envelopes.misc import create_constant_envelope
from sinethesizer.io.events_to_wav import (
    add_event_to_timeline, convert_events_to_timeline, write_timeline_to_wav
)
from sinethesizer.synth.core import Event, Instrument, ModulatedWave, Partial
from sinethesizer.synth.event_to_amplitude_factor import (
    compute_amplitude_factor_as_power_of_velocity
)


@pytest.mark.parametrize(
    "timeline, event, instruments_registry, frame_rate, expected",
    [
        (
            # `timeline`
            np.array([
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, -4, -5, -6, -7, -8, -9]
            ]),
            # `event`
            Event(
                instrument='sine',
                start_time=2,
                duration=1,
                frequency=1,
                velocity=1,
                effects='',
                frame_rate=4
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
                                quasiperiodic_bandwidth=0
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
            # `frame_rate`
            4,
            # `expected`
            np.array([
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, -1],
                [1, 2, 3, -4, -5, -6, -7, -8, -9, 1, 0, -1]
            ])
        ),
    ]
)
def test_add_event_to_timeline(
        timeline: np.ndarray, event: Event,
        instruments_registry: Dict[str, Instrument], frame_rate: int,
        expected: np.ndarray
) -> None:
    """Test `add_event_to_timeline` function."""
    result = add_event_to_timeline(
        timeline, event, instruments_registry, frame_rate
    )
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "events, settings, expected",
    [
        (
            # `events`
            [
                Event(
                    instrument='sine',
                    start_time=1.0,
                    duration=1.0,
                    frequency=1.0,
                    velocity=0.5,
                    effects='',
                    frame_rate=4
                ),
                Event(
                    instrument='sine',
                    start_time=2.0,
                    duration=1.0,
                    frequency=1.0,
                    velocity=1.0,
                    effects='',
                    frame_rate=4
                ),
            ],
            # `settings`
            {
                'frame_rate': 4,
                'trailing_silence': 1,
                'instruments_registry': {
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
                                    quasiperiodic_bandwidth=0
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
                }
            },
            # `expected`
            np.array([
                [0, 0, 0, 0, 0, 0.5, 0, -0.5, 0, 1.0, 0, -1.0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.5, 0, -0.5, 0, 1.0, 0, -1.0, 0, 0, 0, 0]
            ])
        ),
    ]
)
def test_convert_events_to_timeline(
        events: List[Event], settings: Dict[str, Any], expected: np.ndarray
) -> None:
    """Test `convert_events_to_timeline` function."""
    result = convert_events_to_timeline(events, settings)
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "timeline, frame_rate",
    [(np.array([[1, 2, 3], [2, 3, 4]]), 10)]
)
def test_write_timeline_to_wav(
        path_to_tmp_file: str, timeline: np.ndarray, frame_rate: int
) -> None:
    """Test `write_timeline_to_wav` function."""
    write_timeline_to_wav(path_to_tmp_file, timeline, frame_rate)
