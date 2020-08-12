"""
Create and modify air pressure timeline.

Author: Nikolay Lysenko
"""


import json
from math import ceil
from typing import Any, Dict, List

import numpy as np

from sinethesizer.effects import get_effects_registry
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
        sound: np.ndarray, sound_info: Dict[str, Any], effects_def: str
) -> np.ndarray:
    """
    Apply sound effects.

    :param sound:
        sound to be modified
    :param sound_info:
        information about `sound` variable such as number of frames per second
        and its fundamental frequency (if it exists)
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
        sound = effects_registry[effect_name](sound, sound_info, **effect)
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
        maximum possible delay between channels in seconds (for Haas effect);
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
    sound_info = {
        'frame_rate': frame_rate,
        'fundamental_frequency': event['frequency']
    }
    sound = apply_effects(sound, sound_info, event['effects'])
    start_frame = ceil(frame_rate * event['start_time'])
    end_frame = start_frame + sound.shape[1]
    if end_frame > timeline.shape[1]:  # Effects like reverb may prolong event.
        n_extra_frames = end_frame - timeline.shape[1]
        padding = np.zeros((timeline.shape[0], n_extra_frames))
        timeline = np.hstack((timeline, padding))
    timeline[:, start_frame:end_frame] += sound
    return timeline
