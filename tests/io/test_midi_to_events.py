"""
Test `sinethesizer.io.midi_to_events` module.

Author: Nikolay Lysenko
"""


from typing import Any

import pretty_midi
import pytest

from sinethesizer.io.midi_to_events import convert_midi_to_events
from sinethesizer.synth.core import Event


@pytest.mark.parametrize(
    "midi_instrument, midi_events, settings, expected",
    [
        (
            # `midi_instrument`
            {'program': 0, 'name': '1'},
            # `midi_events`
            [
                {'start': 1, 'end': 2, 'pitch': 21, 'velocity': 127},
                {'start': 2, 'end': 3, 'pitch': 25, 'velocity': 127},
            ],
            # `settings`
            {
                'frame_rate': 4,
                'trailing_silence': 1,
                'midi': {
                    'program_to_instrument': {0: 'sine'}
                },
            },
            # `expected`
            [
                Event(
                    instrument='sine',
                    start_time=1.0,
                    duration=1.0,
                    frequency=27.5,  # A0
                    velocity=1.0,
                    effects='',
                    frame_rate=4
                ),
                Event(
                    instrument='sine',
                    start_time=2.0,
                    duration=1.0,
                    frequency=34.64782887210901,  # C#1
                    velocity=1.0,
                    effects='',
                    frame_rate=4
                ),
            ]
        ),
        (
            # `midi_instrument`
            {'program': 0, 'name': '1'},
            # `midi_events`
            [
                {'start': 1, 'end': 2, 'pitch': 21, 'velocity': 127},
                {'start': 2, 'end': 3, 'pitch': 25, 'velocity': 127},
            ],
            # `settings`
            {
                'frame_rate': 4,
                'trailing_silence': 1,
                'midi': {
                    'track_name_to_instrument': {'1': 'woodwind'},
                    'track_name_to_effects': {'1': '"name": "artificial_reverb"'},
                    'program_to_instrument': {0: 'sine'},
                    'program_to_effects': {0: '"name": "vibrato"'},
                },
            },
            # `expected`
            [
                Event(
                    instrument='woodwind',
                    start_time=1.0,
                    duration=1.0,
                    frequency=27.5,  # A0
                    velocity=1.0,
                    effects='"name": "artificial_reverb"',
                    frame_rate=4
                ),
                Event(
                    instrument='woodwind',
                    start_time=2.0,
                    duration=1.0,
                    frequency=34.64782887210901,  # C#1
                    velocity=1.0,
                    effects='"name": "artificial_reverb"',
                    frame_rate=4
                ),
            ]
        ),
        (
            # `midi_instrument`
            {'program': 0, 'name': '1'},
            # `midi_events`
            [
                {'start': 1, 'end': 2, 'pitch': 21, 'velocity': 127},
                {'start': 2, 'end': 3, 'pitch': 25, 'velocity': 127},
            ],
            # `settings`
            {
                'frame_rate': 4,
                'trailing_silence': 1,
                'midi': {
                    'track_name_to_instrument': {'2': 'woodwind'},
                    'track_name_to_effects': {'1': '"name": "artificial_reverb"'},
                    'program_to_instrument': {0: 'sine'},
                    'program_to_effects': {0: '"name": "vibrato"'},
                },
            },
            # `expected`
            []
        ),
    ]
)
def test_convert_midi_to_timeline(
        path_to_tmp_file: str, midi_instrument: dict[str, Any], midi_events: list[dict[str, Any]],
        settings: dict[str, Any], expected: list[Event]
) -> None:
    """Test `convert_midi_to_timeline` function."""
    pretty_midi_instrument = pretty_midi.Instrument(**midi_instrument)
    for event in midi_events:
        note = pretty_midi.Note(**event)
        pretty_midi_instrument.notes.append(note)
    composition = pretty_midi.PrettyMIDI()
    composition.instruments.append(pretty_midi_instrument)
    composition.write(path_to_tmp_file)
    result = convert_midi_to_events(path_to_tmp_file, settings)
    assert result == expected
