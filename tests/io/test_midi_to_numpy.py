"""
Test `sinethesizer.io.midi_to_numpy` module.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Any, Dict, List

import numpy as np
import pretty_midi
import pytest

from sinethesizer.io.midi_to_numpy import convert_midi_to_timeline
from sinethesizer.synth.envelopes import trapezoid
from sinethesizer.synth.timbre import TimbreSpec


@pytest.mark.parametrize(
    "midi_events, program, program_to_timbre_content, settings, expected",
    [
        (
            # `midi_events`
            [
                {'start': 1, 'end': 2, 'pitch': 21, 'velocity': 100},
                {'start': 2, 'end': 3, 'pitch': 25, 'velocity': 100},
            ],
            # `program`
            0,
            # `program_to_timbre_content`
            ["0: sine"],
            # `settings`
            {
                'frame_rate': 4,
                'trailing_silence': 1,
                'max_channel_delay': 0.02,
                'timbres_registry': {
                    'sine': TimbreSpec(
                        fundamental_waveform='sine',
                        fundamental_volume_envelope_fn=partial(
                            trapezoid,
                            begin_share=0, end_share=0
                        ),
                        fundamental_effects=[],
                        overtones_specs=[]
                    )
                }
            },
            # `expected`
            np.array([
                [
                    0, 0, 0, 0,
                    0, -0.707106781, -1, -0.707106781,
                    0, -0.85085328, 0.89408235, -0.08865447,
                    0, 0, 0, 0
                ],
                [
                    0, 0, 0, 0,
                    0, -0.707106781, -1, -0.707106781,
                    0, -0.85085328, 0.89408235, -0.08865447,
                    0, 0, 0, 0
                ]
            ])
        ),
    ]
)
def test_convert_midi_to_timeline(
        path_to_tmp_file: str, path_to_another_tmp_file: str,
        midi_events: List[Dict[str, Any]], program: int,
        program_to_timbre_content: str, settings: Dict[str, Any],
        expected: np.ndarray
) -> None:
    """Test `convert_midi_to_timeline` function."""
    pretty_midi_instrument = pretty_midi.Instrument(program)
    for event in midi_events:
        note = pretty_midi.Note(**event)
        pretty_midi_instrument.notes.append(note)
    composition = pretty_midi.PrettyMIDI()
    composition.instruments.append(pretty_midi_instrument)
    composition.write(path_to_tmp_file)
    with open(path_to_another_tmp_file, 'w') as tmp_yaml_file:
        for line in program_to_timbre_content:
            tmp_yaml_file.write(line + '\n')
    result = convert_midi_to_timeline(
        path_to_tmp_file, path_to_another_tmp_file, settings
    )
    np.testing.assert_almost_equal(result, expected)
