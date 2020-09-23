"""
Read MIDI file and convert it to sound events.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import pretty_midi

from sinethesizer.synth.core import Event


MAX_MIDI_VALUE = 127


def convert_midi_to_events(
        midi_path: str, settings: Dict[str, Any]
) -> List[Event]:
    """
    Collect sound events (loosely speaking, played notes) from a MIDI file.

    :param midi_path:
        path to source MIDI file
    :param settings:
        global settings for the track
    :return:
        sound events
    """
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for pretty_midi_instrument in midi_data.instruments:
        program = pretty_midi_instrument.program
        sinethesizer_instrument = settings['midi_mapping'][program]
        for note in pretty_midi_instrument.notes:
            event = Event(
                instrument=sinethesizer_instrument,
                start_time=note.start,
                duration=note.end - note.start,
                frequency=pretty_midi.note_number_to_hz(note.pitch),
                velocity=note.velocity / MAX_MIDI_VALUE,
                effects='',
                frame_rate=settings['frame_rate']
            )
            events.append(event)
    return events
