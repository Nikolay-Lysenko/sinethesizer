"""
Create and modify air pressure timeline.

Author: Nikolay Lysenko
"""


import json
from math import ceil
from typing import Any, Dict, List

import numpy as np

from sinethesizer.synth.effects import get_effects_registry
from sinethesizer.synth.synth import synthesize
from sinethesizer.synth.timbre import TimbreSpec


EVENTS_TYPE = List[Dict[str, Any]]


def create_empty_timeline(
        events: EVENTS_TYPE, frame_rate: int, trailing_silence: float
) -> np.ndarray:
    """
    Create empty timeline of air pressure.

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
        JSON string with list of effects to be applied
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
        it is a measure of size of imaginary space occupied by sound sources
    :param frame_rate:
        number of frames per second
    :return:
        timeline with sound event added
    """
    sound = synthesize(
        timbres_registry[event['timbre']],
        event['frequency'],
        event['volume'],
        event['duration'],
        event['location'],
        max_channel_delay,
        frame_rate
    )
    sound = apply_effects(sound, frame_rate, event['effects'])
    n_frames_before = ceil(frame_rate * event['start_time'])
    timeline[:, n_frames_before:n_frames_before+sound.shape[1]] += sound
    return timeline
