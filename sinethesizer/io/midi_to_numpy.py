"""
Read MIDI file and convert it to pressure timeline.

Author: Nikolay Lysenko
"""


from typing import Any, Dict

import numpy as np
import pretty_midi
import yaml

from sinethesizer.synth.timeline import (
    EVENTS_TYPE, add_event_to_timeline, create_empty_timeline
)


def collect_events(
        midi_path: str, program_to_timbre: Dict[int, str]
) -> EVENTS_TYPE:
    """
    Collect sound events (played notes) from a MIDI file.

    :param midi_path:
        path to source MIDI file
    :param program_to_timbre:
        mapping from instrument ID (according to General MIDI specification)
        to `sinethesizer` timbre
    :return:
        sound events
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for instrument in midi_data.instruments:
        program = instrument.program
        timbre = program_to_timbre[program]
        for note in instrument.notes:
            events.append(
                {
                    'timbre': timbre,
                    'start_time': note.start,
                    'frequency': pretty_midi.note_number_to_hz(note.pitch),
                    'duration': note.end - note.start,
                    'volume': note.velocity / 100,
                    'location': 0,
                    'effects': '',
                }
            )
    return events


def convert_midi_to_timeline(
        midi_path: str, program_to_timbre_path: str, settings: Dict[str, Any]
) -> np.ndarray:
    """
    Create pressure timeline based on TSV file.

    :param midi_path:
        path to MIDI file
    :param program_to_timbre_path:
        path to YAML file with mapping from instrument ID (according to
        General MIDI specification) to `sinethesizer` timbre
    :param settings:
        global settings for the track
    :return:
        sound represented as pressure timeline
    """
    with open(program_to_timbre_path) as mapping_file:
        program_to_timbre = yaml.safe_load(mapping_file)
    events = collect_events(midi_path, program_to_timbre)

    timeline = create_empty_timeline(
        events, settings['frame_rate'], settings['trailing_silence']
    )
    for event in events:
        timeline = add_event_to_timeline(
            timeline, event, settings['timbres_registry'],
            settings['max_channel_delay'], settings['frame_rate']
        )
    return timeline
