"""
Read TSV file of special schema and convert it to pressure timeline.

Author: Nikolay Lysenko
"""


import os
import json
from math import ceil
from typing import List, Dict, Any, Union

import numpy as np

from sinethesizer.synth import synthesize, get_effects_registry
from sinethesizer.synth.timbre import TimbreSpec
from sinethesizer.io.utils import convert_note_to_frequency


def create_empty_timeline(
        events: List[Dict[str, Any]], frame_rate: int, trailing_silence: float
) -> np.ndarray:
    """
    Create empty timeline.

    :param events:
        sound events that should fit to the timeline to be created
    :param frame_rate:
        number of frames per second
    :param trailing_silence:
        number of seconds with silence at the end of the timeline
    :return:
        empty timeline
    """
    max_event_time = max(
        event['start_time'] + event['duration'] for event in events
    )
    duration_in_seconds = max_event_time + trailing_silence
    duration_in_frames = ceil(frame_rate * duration_in_seconds)
    mono_timeline = np.zeros(duration_in_frames)
    timeline = np.vstack((mono_timeline, mono_timeline))
    return timeline


def apply_effects(
        sound: np.ndarray, frame_rate: int, effects_def: str
) -> np.ndarray:
    """
    Apply sound effects.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param effects_def:
        jSON string with list of effects to be applied
    :return:
        modified sound
    """
    if not effects_def:
        return sound
    effects_registry = get_effects_registry()
    effects = json.loads(effects_def)
    for effect in effects:
        effect_name = effect.pop('name')
        sound = effects_registry[effect_name](sound, frame_rate, **effect)
    return sound


def add_event_to_timeline(
        timeline: np.ndarray, event: Dict[str, Any],
        timbres_registry: Dict[str, TimbreSpec],
        max_channel_delay: float, frame_rate: int
) -> np.ndarray:
    """
    Add sound event (say, played note) to timeline.

    :param timeline:
        timeline of pressure deviations
    :param event:
        parameters of sound piece that should be added
    :param timbres_registry:
        mapping from timbre name to its specification
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is a measure of potential size of space occupied by sound sources
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
        timbres_registry[event['timbre']],
        frequency,
        event['volume'],
        event['duration'],
        event['location'],
        max_channel_delay,
        frame_rate
    )
    sound = apply_effects(sound, frame_rate, event['effects'])
    mono_past_padding = np.zeros(ceil(frame_rate * event['start_time']))
    past_padding = np.vstack((mono_past_padding, mono_past_padding))
    n_frames_left = timeline.shape[1] - sound.shape[1] - past_padding.shape[1]
    mono_future_padding = np.zeros(n_frames_left)
    future_padding = np.vstack((mono_future_padding, mono_future_padding))
    sound = np.concatenate((past_padding, sound, future_padding), axis=1)
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
        schema = input_file.readline().rstrip(os.linesep).split('\t')
        for line in input_file.readlines():
            events.append(
                dict(zip(schema, line.rstrip(os.linesep).split('\t')))
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
