"""
Read TSV file of special schema and convert it to pressure timeline.

Author: Nikolay Lysenko
"""


from math import ceil
from typing import List, Tuple, Dict, Any, Union

import numpy as np

from sinethesizer.synth import synthesize
from sinethesizer.presets import TIMBRES_REGISTRY
from sinethesizer.io.utils import convert_note_to_frequency


def create_empty_timeline(
        events: List[Dict[str, Any]], frame_rate: int, tail_silence: float
) -> np.ndarray:
    """
    Create empty timeline.

    :param events:
        sound events that should fit to the timeline to be created
    :param frame_rate:
        number of frames per second
    :param tail_silence:
        number of seconds with silence at the end of the timeline
    :return:
        empty timeline
    """
    max_event_time = max(
        event['start_time'] + event['duration'] for event in events
    )
    duration_in_seconds = max_event_time + tail_silence
    duration_in_frames = ceil(frame_rate * duration_in_seconds)
    timeline = np.zeros(duration_in_frames)
    return timeline


def add_event_to_timeline(
        timeline: np.ndarray, event: Dict[str, Any], frame_rate: int
) -> np.ndarray:
    """
    Add sound event (say, played note) to timeline.

    :param timeline:
        timeline of pressure deviations
    :param event:
        parameters of sound piece that should be added
    :param frame_rate:
        number of frames per second
    :return:
        timeline with sound event added
    """
    if isinstance(event['frequency'], str):
        frequency = convert_note_to_frequency(event['frequency'])
    else:
        frequency = event['frequency']
    sound = synthesize(
        TIMBRES_REGISTRY[event['timbre']],
        frequency,
        event['volume'],
        event['duration'],
        frame_rate
    )
    left_padding = np.zeros(ceil(frame_rate * event['start_time']))
    right_padding = np.zeros(len(timeline) - len(sound) - len(left_padding))
    sound = np.concatenate((left_padding, sound, right_padding))
    timeline += sound
    return timeline


def set_types(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Set types of parsed from TSV file fields.

    :param events:
        sound events where all values have type `str`
    :return:
        sound events where values have proper types
    """

    def maybe_convert_to_float(x: str) -> Union[str, float]:
        try:
            x = float(x)
        except ValueError:
            pass
        return x

    field_to_caster = {
        'start_time': float,
        'duration': float,
        'frequency': maybe_convert_to_float,
        'volume': float,
        'location': float
    }
    events = [
        {k: field_to_caster.get(k, lambda x: x)(v) for k, v in event.items()}
        for event in events
    ]
    return events


def convert_tsv_to_timeline(
        input_path: str, frame_rate: int, tail_silence: float
) -> Tuple[np.ndarray, int]:
    """
    Create pressure timeline based on TSV file.

    :param input_path:
        path to JSON file of special schema
    :param frame_rate:
        number of frames per second
    :param tail_silence:
        number of seconds with silence at the end of the timeline
    :return:
        sound represented as pressure timeline and its frame rate
    """
    events = []
    with open(input_path) as input_file:
        schema = input_file.readline().split('\t')
        for line in input_file.readlines():
            events.append(dict(zip(schema, line.split('\t'))))
    events = set_types(events)

    timeline = create_empty_timeline(events, frame_rate, tail_silence)
    for event in events:
        timeline = add_event_to_timeline(timeline, event, frame_rate)
    return timeline, frame_rate
