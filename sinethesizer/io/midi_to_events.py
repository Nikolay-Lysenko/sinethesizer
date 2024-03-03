"""
Read MIDI file and convert it to sound events.

Author: Nikolay Lysenko
"""


from typing import Any

import pretty_midi

from sinethesizer.synth.core import Event


MAX_MIDI_VALUE = 127


def convert_midi_to_events(
        midi_path: str, settings: dict[str, Any]
) -> list[Event]:
    """
    Collect sound events (loosely speaking, played notes) from a MIDI file.

    :param midi_path:
        path to source MIDI file
    :param settings:
        global settings for the output track
    :return:
        sound events
    """
    midi_settings = settings['midi']
    if 'track_name_to_instrument' in midi_settings:
        instruments_mapping = midi_settings['track_name_to_instrument']
        effects_mapping = midi_settings.get('track_name_to_effects', {})
        key_fn = lambda instrument: instrument.name
    elif 'program_to_instrument' in midi_settings:
        instruments_mapping = midi_settings['program_to_instrument']
        effects_mapping = midi_settings.get('program_to_effects', {})
        key_fn = lambda instrument: instrument.program
    else:
        raise RuntimeError("MIDI config file lacks required sections.")

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for pretty_midi_instrument in midi_data.instruments:
        key = key_fn(pretty_midi_instrument)
        sinethesizer_instrument = instruments_mapping.get(key)
        if sinethesizer_instrument is None:
            continue
        for note in pretty_midi_instrument.notes:
            event = Event(
                instrument=sinethesizer_instrument,
                start_time=note.start,
                duration=note.end - note.start,
                frequency=pretty_midi.note_number_to_hz(note.pitch),
                velocity=note.velocity / MAX_MIDI_VALUE,
                effects=effects_mapping.get(key, ''),
                frame_rate=settings['frame_rate']
            )
            events.append(event)
    return events
