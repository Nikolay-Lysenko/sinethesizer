"""
Read TSV file of special schema and convert it to pressure timeline.

Author: Nikolay Lysenko
"""


import os
from typing import Any, Dict

import numpy as np

from sinethesizer.io.utils import convert_note_to_frequency
from sinethesizer.synth.timeline import (
    EVENTS_TYPE, add_event_to_timeline, create_empty_timeline
)


def set_types(events: EVENTS_TYPE) -> EVENTS_TYPE:
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
        'volume': float,
        'location': float
    }
    events = [
        {k: field_to_caster.get(k, lambda x: x)(v) for k, v in event.items()}
        for event in events
    ]
    return events


def convert_tsv_to_timeline(
        input_path: str, settings: Dict[str, Any]
) -> np.ndarray:
    """
    Create pressure timeline based on TSV file.

    :param input_path:
        path to TSV file with rows representing events
    :param settings:
        global settings for the track
    :return:
        sound represented as pressure timeline
    """
    events = []
    with open(input_path) as input_file:
        column_names = input_file.readline().rstrip(os.linesep).split('\t')
        for line in input_file.readlines():
            events.append(
                dict(zip(column_names, line.rstrip(os.linesep).split('\t')))
            )
    events = set_types(events)

    timeline = create_empty_timeline(
        events, settings['frame_rate'], settings['trailing_silence']
    )
    for event in events:
        timeline = add_event_to_timeline(
            timeline, event, settings['timbres_registry'],
            settings['max_channel_delay'], settings['frame_rate']
        )
    return timeline
