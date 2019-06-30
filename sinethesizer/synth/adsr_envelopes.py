"""
Define ADSR (Attack-Decay-Sustain-Release) envelopes.

Author: Nikolay Lysenko
"""


from math import ceil
from typing import Callable, Dict

import numpy as np


ENVELOPE_FN_TYPE = Callable[[float, int], np.ndarray]


def relative_adsr(
        duration: float, frame_rate: int,
        attack_share: float = 0.15, decay_share: float = 0.15,
        sustain_level: float = 0.6, release_share: float = 0.2
) -> np.ndarray:
    """
    Create envelope with attack, decay, and release proportional to duration.

    :param duration:
        duration of sound in seconds
    :param frame_rate:
        number of frames per second
    :param attack_share:
        share of attack stage
    :param decay_share:
        share of decay stage
    :param sustain_level:
        volume level at sustain stage where 1 is the peak level
    :param release_share:
        share of release stage
    :return:
        envelope
    """
    duration_in_frames = ceil(duration * frame_rate)

    if attack_share > 0:
        n_frames_with_attack = int(round(attack_share * duration_in_frames))
        step = 1 / n_frames_with_attack
        attack = np.arange(0, 1, step)
    else:
        attack = np.array([])

    if decay_share > 0:
        n_frames_with_decay = int(round(decay_share * duration_in_frames))
        step = (1 - sustain_level) / n_frames_with_decay
        decay = np.arange(1, sustain_level, -step)
    else:
        decay = np.array([])

    if release_share > 0:
        n_frames_with_release = int(round(release_share * duration_in_frames))
        step = sustain_level / n_frames_with_release
        release = np.arange(sustain_level, 0, -step)
    else:
        release = np.array([])

    n_frames_with_sustain = (
        duration_in_frames - len(attack) - len(decay) - len(release)
    )
    sustain = sustain_level * np.ones(n_frames_with_sustain)

    envelope = np.concatenate((attack, decay, sustain, release))
    return envelope


def absolute_adsr(
        duration: float, frame_rate: int,
        attack_time: float = 0.3, decay_time: float = 0.2,
        sustain_level: float = 0.6, release_time: float = 0.49
) -> np.ndarray:
    """
    Create envelope with fixed maximum times of attack, decay, and release.

    :param duration:
        duration of sound in seconds
    :param frame_rate:
        number of frames per second
    :param attack_time:
        maximum attack time in seconds
    :param decay_time:
        maximum decay time in seconds
    :param sustain_level:
        volume level at sustain stage where 1 is the peak level
    :param release_time:
        maximum release time in seconds
    :return:
        envelope
    """
    duration_in_frames = ceil(duration * frame_rate)

    max_n_frames_with_attack = int(round(attack_time * frame_rate))
    max_n_frames_with_decay = int(round(decay_time * frame_rate))
    max_n_frames_with_release = int(round(release_time * frame_rate))

    adr_duration_in_frames = (
        max_n_frames_with_attack
        + max_n_frames_with_decay
        + max_n_frames_with_release
    )
    sustain_duration_in_frames = duration_in_frames - adr_duration_in_frames

    if sustain_duration_in_frames < 0:
        envelope = relative_adsr(
            duration, frame_rate,
            max_n_frames_with_attack / adr_duration_in_frames,
            max_n_frames_with_decay / adr_duration_in_frames,
            sustain_level,
            max_n_frames_with_release / adr_duration_in_frames
        )
        return envelope

    if max_n_frames_with_attack > 0:
        step = 1 / max_n_frames_with_attack
        attack = np.arange(0, 1, step)
    else:
        attack = np.array([])
    if max_n_frames_with_decay > 0:
        step = (1 - sustain_level) / max_n_frames_with_decay
        decay = np.arange(1, sustain_level, -step)
    else:
        decay = np.array([])
    if max_n_frames_with_release > 0:
        step = sustain_level / max_n_frames_with_release
        release = np.arange(sustain_level, 0, -step)
    else:
        release = np.array([])
    n_frames_with_sustain = (
        duration_in_frames - len(attack) - len(decay) - len(release)
    )
    sustain = sustain_level * np.ones(n_frames_with_sustain)

    envelope = np.concatenate((attack, decay, sustain, release))
    return envelope


def spike(
        duration: float, frame_rate: int, breakpoint_location: float = 0.2
) -> np.ndarray:
    """
    Create envelope with amplitude that grows linearly and then falls linearly.

    :param duration:
        duration of sound in frames
    :param frame_rate:
        number of frames per second
    :param breakpoint_location:
        share of frames that are before the breakpoint
    :return:
        envelope
    """
    envelope = relative_adsr(
        duration, frame_rate,
        attack_share=breakpoint_location,
        decay_share=(1 - breakpoint_location),
        sustain_level=0,
        release_share=0
    )
    return envelope


def constant_with_linear_ends(
        duration: float, frame_rate: int,
        begin_share: float = 0.1, end_share: float = 0.1
) -> np.ndarray:
    """
    Create envelope with amplitude that is constant everywhere except its ends.

    :param duration:
        duration of sound in seconds
    :param frame_rate:
        number of frames per second
    :param begin_share:
        share of first frames where amplitude increases linearly
    :param end_share:
        share of last frames where amplitude decreases linearly
    :return:
        envelope
    """
    envelope = relative_adsr(
        duration, frame_rate,
        attack_share=begin_share,
        decay_share=0,
        sustain_level=1,
        release_share=end_share
    )
    return envelope


def get_envelopes_registry() -> Dict[str, ENVELOPE_FN_TYPE]:
    """
    Get mapping from envelope names to functions that create them.

    :return:
        registry of effects
    """
    registry = {
        'absolute_adsr': absolute_adsr,
        'relative_adsr': relative_adsr,
        'spike': spike,
        'constant_with_linear_ends': constant_with_linear_ends
    }
    return registry
