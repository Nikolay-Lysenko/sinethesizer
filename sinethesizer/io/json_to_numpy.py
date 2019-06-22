"""
Read JSON file of special schema and convert it to pressure timeline.

Author: Nikolay Lysenko
"""


import json
from typing import Dict, Any

import numpy as np

from sinethesizer.synth import synthesize
from sinethesizer.presets import TIMBRES_REGISTRY


def create_empty_timeline(input_data: Dict[str, Any]) -> np.ndarray:
    """
    Create empty timeline.

    :param input_data:
        parsed content of input file
    :return:
        empty timeline
    """
    max_event_time = max(
        event['start_time'] + event['duration']
        for event in input_data['events']
    )
    duration_in_seconds = max_event_time + input_data['silence_at_the_end']
    duration_in_frames = input_data['frame_rate'] * duration_in_seconds
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
    left_padding = np.zeros(frame_rate * event['start_time'])
    max_time = event['start_time'] + event['duration']
    right_padding = np.zeros(len(timeline) - frame_rate * max_time)
    sound = synthesize(
        TIMBRES_REGISTRY[event['timbre']],
        event['frequency'],
        event['volume'],
        event['duration'],
        frame_rate
    )
    sound = np.concatenate((left_padding, sound, right_padding))
    timeline += sound
    return timeline


def convert_json_to_timeline(input_path: str) -> np.ndarray:
    """
    Create pressure timeline based on JSON file.

    :param input_path:
        path to JSON file of special schema
    :return:
        sound represented as pressure timeline
    """
    with open(input_path) as input_file:
        input_data = json.load(input_file)
    timeline = create_empty_timeline(input_data)
    frame_rate = input_data['frame_rate']
    for event in input_data['events']:
        timeline = add_event_to_timeline(timeline, event, frame_rate)
    return timeline
