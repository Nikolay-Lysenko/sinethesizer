"""
Read TSV file of special schema and convert it to sound events.

Author: Nikolay Lysenko
"""


import os
from typing import Any, Dict, List

from sinethesizer.synth.core import Event
from sinethesizer.utils.music_theory import convert_note_to_frequency


def set_types(events: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Set types of parsed from TSV file fields.

    :param events:
        sound events where all values have type `str`
    :return:
        sound events where values have proper types
    """

    def parse_frequency(x: str) -> float:
        try:
            x = float(x)
        except ValueError:
            x = convert_note_to_frequency(x)
        return x

    field_to_caster = {
        'start_time': float,
        'duration': float,
        'frequency': parse_frequency,
        'velocity': float
    }
    events = [
        {k: field_to_caster.get(k, lambda x: x)(v) for k, v in event.items()}
        for event in events
    ]
    return events


def convert_tsv_to_events(
        input_path: str, settings: Dict[str, Any]
) -> List[Event]:
    """
    Collect sound events (loosely speaking, played notes) from a TSV file.

    :param input_path:
        path to TSV file with rows representing events
    :param settings:
        global settings for the track
    :return:
        sound events
    """
    raw_events = []
    with open(input_path) as input_file:
        column_names = input_file.readline().rstrip(os.linesep).split('\t')
        for line in input_file.readlines():
            raw_events.append(
                dict(zip(column_names, line.rstrip(os.linesep).split('\t')))
            )
    raw_events = set_types(raw_events)

    events = []
    fields_to_use = [
        'instrument',
        'start_time',
        'duration',
        'frequency',
        'velocity',
        'effects'
    ]
    for raw_event in raw_events:
        raw_event = {k: v for k, v in raw_event.items() if k in fields_to_use}
        event = Event(frame_rate=settings['frame_rate'], **raw_event)
        events.append(event)
    return events
