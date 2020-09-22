"""
Convert events to array with pressure deviations timeline and to WAV file.

Author: Nikolay Lysenko
"""


from math import ceil
from typing import Any, Dict, List

import numpy as np
import scipy.io.wavfile

from sinethesizer.synth.core import Event, Instrument, synthesize


def create_empty_timeline(
        events: List[Event], frame_rate: int, trailing_silence: float
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
    max_event_time = max(event.start_time + event.duration for event in events)
    duration_in_seconds = max_event_time + trailing_silence
    duration_in_frames = ceil(frame_rate * duration_in_seconds)
    mono_timeline = np.zeros(duration_in_frames)
    timeline = np.vstack((mono_timeline, mono_timeline))
    return timeline


def add_event_to_timeline(
        timeline: np.ndarray, event: Event,
        instruments_registry: Dict[str, Instrument], frame_rate: int
) -> np.ndarray:
    """
    Add sound event to timeline.

    :param timeline:
        timeline of pressure deviations
    :param event:
        parameters of sound event that should be added
    :param instruments_registry:
        mapping from instrument name to its representation
    :param frame_rate:
        number of frames per second
    :return:
        timeline with sound event added
    """
    sound = synthesize(event, instruments_registry)
    start_frame = ceil(frame_rate * event.start_time)
    end_frame = start_frame + sound.shape[1]
    if end_frame > timeline.shape[1]:  # Effects like reverb may prolong event.
        n_extra_frames = end_frame - timeline.shape[1]
        padding = np.zeros((timeline.shape[0], n_extra_frames))
        timeline = np.hstack((timeline, padding))
    timeline[:, start_frame:end_frame] += sound
    return timeline


def convert_events_to_timeline(
        events: List[Event], settings: Dict[str, Any]
) -> np.ndarray:
    """
    Convert events to array with pressure deviations timeline.

    :param events:
        sound events
    :param settings:
        global settings for the track
    :return:
        pressure deviations timeline
    """
    timeline = create_empty_timeline(
        events, settings['frame_rate'], settings['trailing_silence']
    )
    for event in events:
        timeline = add_event_to_timeline(
            timeline, event, settings['instruments_registry'],
            settings['frame_rate']
        )
    return timeline


def write_timeline_to_wav(
        output_path: str, timeline: np.ndarray, frame_rate: int
) -> None:
    """
    Write pressure deviations timeline to WAV file.

    :param output_path:
        path to resulting file
    :param timeline:
        sound represented as pressure deviations timeline
    :param frame_rate:
        number of frames per second
    :return:
        None
    """
    scipy.io.wavfile.write(output_path, frame_rate, timeline.T)
