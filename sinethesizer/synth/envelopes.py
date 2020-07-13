"""
Define envelopes for wave amplitude.

Supported types of envelopes are as follows:
* ADSR (Attack-Decay-Sustain-Release);
* user-defined.

Author: Nikolay Lysenko
"""


from math import ceil, floor
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.signal import upfirdn


ENVELOPE_FN_TYPE = Callable[[float, int], np.ndarray]


def generic_adsr(
        duration: float, frame_rate: int,
        attack_to_adsr_max_ratio: float = 0.15,
        max_attack_duration: float = 0.25,
        attack_degree: float = 1.0,
        decay_to_dsr_max_ratio: float = 0.28,
        max_decay_duration: float = 0.25,
        decay_degree: float = 1.0,
        sustain_level: float = 0.6,
        max_sustain_duration: Optional[float] = None,
        release_to_sr_approx_ratio: float = 0.5,
        max_release_duration: Optional[float] = 1.0,
        release_degree: float = 1.0
) -> np.ndarray:
    """
    Create envelope of shape that depends on numerous parameters.

    :param duration:
        duration of sound in seconds
    :param frame_rate:
        number of frames per second
    :param attack_to_adsr_max_ratio:
        maximum fraction of frames with attack amongst all frames (i.e.,
        frames with attack, decay, sustain, and release)
    :param max_attack_duration:
        maximum duration of attack in seconds
    :param attack_degree:
        degree of attack dynamic; if it is 1, attack is linear; if it is
        greater than 1, attack is concave; if it is less than 1,
        attack is convex
    :param decay_to_dsr_max_ratio:
        maximum fraction of frames with decay amongst frames with decay,
        sustain, and release
    :param max_decay_duration:
        maximum duration of decay in seconds
    :param decay_degree:
        degree of decay dynamic; if it is 1, decay is linear; if it is
        greater than 1, decay is concave; if it is less than 1,
        decay is convex
    :param sustain_level:
        volume level at sustain stage where 1 is the peak level
        (i.e., level at the end of attack)
    :param max_sustain_duration:
        maximum duration of sustain in seconds
    :param release_to_sr_approx_ratio:
        fraction of frames with release amongst all frames with sustain and
        release, but only if resulting durations of sustain and release are
        not clipped due to `max_sustain_duration` or `max_release_duration`;
        else this argument affects nothing
    :param max_release_duration:
        maximum duration of release in seconds
    :param release_degree:
        degree of release dynamic; if it is 1, release is linear; if it is
        greater than 1, release is concave; if it is less than 1,
        release is convex
    :return:
        envelope
    """
    remaining_duration_in_frames = ceil(duration * frame_rate)

    n_frames_with_attack = min(
        floor(attack_to_adsr_max_ratio * remaining_duration_in_frames),
        floor(max_attack_duration * frame_rate)
    )
    if n_frames_with_attack > 0:
        step = 1 / n_frames_with_attack
        xs = np.arange(1, 0, -step)
        attack = 1 - xs ** attack_degree
    else:
        attack = np.array([])
    remaining_duration_in_frames -= len(attack)

    n_frames_with_decay = min(
        floor(decay_to_dsr_max_ratio * remaining_duration_in_frames),
        floor(max_decay_duration * frame_rate)
    )
    if n_frames_with_decay > 0:
        step = 1 / n_frames_with_decay
        xs = np.arange(0, 1, step)
        decay = 1 - (1 - sustain_level) * xs ** decay_degree
    else:
        decay = np.array([])
    remaining_duration_in_frames -= len(decay)

    if max_sustain_duration is None:
        max_sustain_duration = 1e7
    max_n_frames_with_sustain = floor(max_sustain_duration * frame_rate)
    n_frames_with_sustain = min(
        floor((1 - release_to_sr_approx_ratio) * remaining_duration_in_frames),
        max_n_frames_with_sustain
    )
    if max_release_duration is None:
        max_release_duration = 1e7
    max_n_frames_with_release = floor(max_release_duration * frame_rate)
    n_frames_with_release = min(
        remaining_duration_in_frames - n_frames_with_sustain,
        max_n_frames_with_release
    )
    n_frames_with_sustain = min(
        remaining_duration_in_frames - n_frames_with_release,
        max_n_frames_with_sustain
    )
    sustain = sustain_level * np.ones(n_frames_with_sustain)
    remaining_duration_in_frames -= len(sustain)

    if n_frames_with_release > 0:
        step = 1 / n_frames_with_release
        xs = np.arange(0, 1, step)
        release = sustain_level * (1 - xs ** release_degree)
        release = release[:n_frames_with_release]
    else:
        release = np.array([])
    remaining_duration_in_frames -= len(release)

    zero_padding = np.zeros(remaining_duration_in_frames)
    envelope = np.concatenate((attack, decay, sustain, release, zero_padding))
    return envelope


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
        (i.e., level at the end of attack)
    :param release_share:
        share of release stage
    :return:
        envelope
    """
    duration_in_frames = ceil(duration * frame_rate)

    if attack_share > 0:
        n_frames_with_attack = floor(attack_share * duration_in_frames)
        step = 1 / n_frames_with_attack
        attack = np.arange(0, 1, step)
    else:
        attack = np.array([])

    if decay_share > 0:
        n_frames_with_decay = floor(decay_share * duration_in_frames)
        step = (1 - sustain_level) / n_frames_with_decay
        decay = np.arange(1, sustain_level, -step)
    else:
        decay = np.array([])

    if release_share > 0:
        n_frames_with_release = floor(release_share * duration_in_frames)
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


def trapezoid(
        duration: float, frame_rate: int,
        begin_share: float = 0.2, end_share: float = 0.1
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


def user_defined_envelope(
        duration: float, frame_rate: int, parts: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Create envelope that is an upsampled version of a user-defined envelope.

    :param duration:
        duration of sound in seconds
    :param frame_rate:
        number of frames per second
    :param parts:
        list of dictionaries representing successive parts of an envelope;
        there, 'values' key relates to envelope values (from its very
        beginning up to its very end inclusively) and 'max_duration' key
        relates to maximum allowed duration of this part in seconds
    :return:
        envelope
    """
    remaining_duration_in_frames = ceil(duration * frame_rate)
    # 1 is subtracted, because there are N - 1 intervals between N points.
    remaining_length = sum(len(part['values']) - 1 for part in parts)
    results = []
    for part in parts:
        current_length = len(part['values']) - 1
        length_fraction = current_length / remaining_length
        part_duration_in_frames = min(
            floor(length_fraction * remaining_duration_in_frames),
            floor((part['max_duration'] or 1e7) * frame_rate)
        )

        current_result = np.zeros(part_duration_in_frames)
        upsampling_ratio = (part_duration_in_frames - 1) / current_length
        for i, value in enumerate(part['values']):
            index = int(round(i * upsampling_ratio))
            current_result[index] = value

        step = 1 / (2 * round(upsampling_ratio))
        convolution = 1 - 2 * np.abs(np.arange(0, 1, step) - 0.5)[1:]
        current_result = np.convolve(current_result, convolution, mode='same')
        results.append(current_result)
        remaining_length -= current_length
        remaining_duration_in_frames -= len(current_result)

    zero_padding = np.zeros(remaining_duration_in_frames)
    results.append(zero_padding)
    envelope = np.concatenate(results)
    return envelope


def get_envelopes_registry() -> Dict[str, ENVELOPE_FN_TYPE]:
    """
    Get mapping from envelope names to functions that create them.

    :return:
        registry of envelopes
    """
    registry = {
        'generic_adsr': generic_adsr,
        'relative_adsr': relative_adsr,
        'spike': spike,
        'trapezoid': trapezoid,
        'user_defined': user_defined_envelope,
    }
    return registry
