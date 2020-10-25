"""
Define additional envelopes.

Author: Nikolay Lysenko
"""


from math import ceil, floor
from typing import Optional

import numpy as np


def create_constant_envelope(
        event: 'sinethesizer.synth.core.Event', value: float
) -> np.ndarray:
    """
    Create constant envelope.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and velocity
    :param value:
        value of the envelope
    :return:
        constant envelope
    """
    return value * np.ones(ceil(event.duration * event.frame_rate))


def create_exponentially_decaying_envelope(
        event: 'sinethesizer.synth.core.Event',
        attack_to_ad_max_ratio: float = 0.025,
        max_attack_duration: float = 0.025,
        attack_degree: float = 1.0,
        decay_half_life: Optional[float] = None,
        decay_half_life_ratio: Optional[float] = 0.125,
        max_release_duration: float = 0.01,
        release_duration_on_velocity_order: float = 0.0,
        release_degree: float = 1.0,
        peak_value: float = 1.0,
        ratio_at_zero_velocity: float = 0.0,
        envelope_values_on_velocity_order: float = 0.0
) -> np.ndarray:
    """
    Create ADR (Attack-Decay-Release) envelope with exponential decay.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and velocity
    :param attack_to_ad_max_ratio:
        maximum fraction of frames with attack amongst frames with attack
        and decay
    :param max_attack_duration:
        maximum duration of attack in seconds
    :param attack_degree:
        degree of attack dynamic; if it is 1, attack is linear; if it is
        greater than 1, attack is concave; if it is less than 1,
        attack is convex
    :param decay_half_life:
        half-life of decay in seconds; if this argument is passed,
        `decay_half_life_ratio` is ignored
    :param decay_half_life_ratio:
        half-life of decay as ratio to decay duration; this argument is used
        only if decay_half_life is not `None`
    :param max_release_duration:
        maximum duration of release in seconds
    :param release_duration_on_velocity_order:
        coefficient that determines actual duration of release depending on
        velocity; given non-maximum positive velocity, the higher it is,
        the shorter release is; if it is 0, velocity does not affect
        release duration
    :param release_degree:
        degree of release dynamic; if it is 1, release is linear; if it is
        greater than 1, release is concave; if it is less than 1,
        release is convex
    :param peak_value:
        peak envelope value given maximum velocity; usually, this argument
        should be passed only if output envelope is used as modulation
        index envelope
    :param ratio_at_zero_velocity:
        ratio of envelope values at zero velocity to envelope values at maximum
        velocity; usually, this argument should be passed only if output
        envelope is used as modulation index envelope
    :param envelope_values_on_velocity_order:
        coefficient that determines dependence of envelope values on velocity;
        given non-maximum positive velocity, the higher it is, the lower
        envelope values are; if it is 0, velocity does not affect envelope;
        usually, this argument should be passed only if output envelope is
        used as modulation index envelope
    :return:
        envelope
    """
    frame_rate = event.frame_rate
    ad_duration_in_frames = ceil(event.duration * frame_rate)

    n_frames_with_attack = min(
        floor(attack_to_ad_max_ratio * ad_duration_in_frames),
        floor(max_attack_duration * frame_rate)
    )
    if n_frames_with_attack > 0:
        xs = np.linspace(1, 0, n_frames_with_attack)
        attack = 1 - xs ** attack_degree
    else:
        attack = np.array([])
    decay_duration_in_frames = ad_duration_in_frames - len(attack)

    if decay_half_life is not None:
        half_life_in_frames = decay_half_life * frame_rate
    else:
        half_life_in_frames = decay_half_life_ratio * decay_duration_in_frames
    xs = np.linspace(0, decay_duration_in_frames, decay_duration_in_frames)
    decay = 2 ** -(xs / half_life_in_frames)

    release_ratio = event.velocity ** release_duration_on_velocity_order
    release_duration = release_ratio * max_release_duration
    n_frames_with_release = floor(release_duration * frame_rate)
    if n_frames_with_release > 0:
        xs = np.linspace(0, 1, n_frames_with_release)
        release = decay[-1] * (1 - xs ** release_degree)
    else:
        release = np.array([])

    envelope = np.concatenate((attack, decay, release))
    envelope *= peak_value
    coef = event.velocity ** envelope_values_on_velocity_order
    envelope *= ratio_at_zero_velocity + coef * (1 - ratio_at_zero_velocity)
    return envelope
