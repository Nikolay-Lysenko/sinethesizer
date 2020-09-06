"""
Define AHDSR (Attack-Hold-Decay-Sustain-Release) envelopes.

Author: Nikolay Lysenko
"""


from math import ceil, floor

import numpy as np


def generic_ahdsr(
        task: 'sinethesizer.synth.core.Task',
        attack_to_ahds_max_ratio: float = 0.2,
        max_attack_duration: float = 0.2,
        attack_degree: float = 1.0,
        hold_to_hds_max_ratio: float = 0.05,
        max_hold_duration: float = 0.05,
        decay_to_ds_max_ratio: float = 0.25,
        max_decay_duration: float = 0.4,
        decay_degree: float = 1.0,
        sustain_level: float = 0.6,
        max_sustain_duration: float = 10.0,
        max_release_duration: float = 0.5,
        release_sensitivity_to_velocity: float = 0.0,
        release_degree: float = 1.0
) -> np.ndarray:
    """
    Create AHDSR envelope of shape that depends on numerous parameters.

    :param task:
        parameters of sound synthesis task that triggered generation
        of this envelope; it provides information about duration, frame rate,
        and velocity
    :param attack_to_ahds_max_ratio:
        maximum fraction of frames with attack amongst frames with attack,
        hold, decay, and sustain
    :param max_attack_duration:
        maximum duration of attack in seconds
    :param attack_degree:
        degree of attack dynamic; if it is 1, attack is linear; if it is
        greater than 1, attack is concave; if it is less than 1,
        attack is convex
    :param hold_to_hds_max_ratio:
        maximum fraction of frames with hold amongst frames with hold, decay,
        and sustain
    :param max_hold_duration:
        maximum duration of hold in seconds
    :param decay_to_ds_max_ratio:
        maximum fraction of frames with decay amongst frames with decay and
        sustain
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
    :param max_release_duration:
        maximum duration of release in seconds
    :param release_sensitivity_to_velocity:
        coefficient that determines actual duration of release depending on
        velocity; the higher it is, the shorter release is given non-maximum
        velocity; if it is 0, velocity does not affect release duration
    :param release_degree:
        degree of release dynamic; if it is 1, release is linear; if it is
        greater than 1, release is concave; if it is less than 1,
        release is convex
    :return:
        envelope
    """
    frame_rate = task.frame_rate
    remaining_duration_in_frames = ceil(task.duration * frame_rate)

    n_frames_with_attack = min(
        floor(attack_to_ahds_max_ratio * remaining_duration_in_frames),
        floor(max_attack_duration * frame_rate)
    )
    if n_frames_with_attack > 0:
        step = 1 / n_frames_with_attack
        xs = np.arange(1, 0, -step)
        xs = np.clip(xs, 0, None)
        attack = 1 - xs ** attack_degree
    else:
        attack = np.array([])
    remaining_duration_in_frames -= len(attack)

    n_frames_with_hold = min(
        floor(hold_to_hds_max_ratio * remaining_duration_in_frames),
        floor(max_hold_duration * frame_rate)
    )
    hold = np.ones(n_frames_with_hold)
    remaining_duration_in_frames -= len(hold)

    n_frames_with_decay = min(
        floor(decay_to_ds_max_ratio * remaining_duration_in_frames),
        floor(max_decay_duration * frame_rate)
    )
    if n_frames_with_decay > 0:
        step = 1 / n_frames_with_decay
        xs = np.arange(0, 1, step)
        decay = 1 - (1 - sustain_level) * xs ** decay_degree
    else:
        decay = np.array([])
    remaining_duration_in_frames -= len(decay)

    n_frames_with_sustain = min(
        remaining_duration_in_frames,
        floor(max_sustain_duration * frame_rate)
    )
    sustain = sustain_level * np.ones(n_frames_with_sustain)

    release_duration_ratio = task.velocity ** release_sensitivity_to_velocity
    release_duration = release_duration_ratio * max_release_duration
    n_frames_with_release = floor(release_duration * frame_rate)
    if n_frames_with_release > 0:
        step = 1 / n_frames_with_release
        xs = np.arange(0, 1, step)
        release = sustain_level * (1 - xs ** release_degree)
        release = release[:n_frames_with_release]
    else:
        release = np.array([])

    envelope = np.concatenate((attack, hold, decay, sustain, release))
    return envelope
