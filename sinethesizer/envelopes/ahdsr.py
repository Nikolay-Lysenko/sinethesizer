"""
Define AHDSR (Attack-Hold-Decay-Sustain-Release) envelopes.

Author: Nikolay Lysenko
"""


from math import ceil, floor

import numpy as np


def create_generic_ahdsr_envelope(
        event: 'sinethesizer.synth.core.Event',
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
        release_duration_on_velocity_order: float = 0.0,
        release_degree: float = 1.0,
        peak_value: float = 1.0,
        ratio_at_zero_velocity: float = 0.0,
        envelope_values_on_velocity_order: float = 0.0
) -> np.ndarray:
    """
    Create AHDSR envelope of shape that depends on numerous parameters.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
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
        amplitude level at sustain stage; a float between 0 and 1 where 1 is
        the peak level (i.e., level at the end of attack)
    :param max_sustain_duration:
        maximum duration of sustain in seconds
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
    remaining_duration_in_frames = ceil(event.duration * frame_rate)

    n_frames_with_attack = min(
        floor(attack_to_ahds_max_ratio * remaining_duration_in_frames),
        floor(max_attack_duration * frame_rate)
    )
    if n_frames_with_attack > 0:
        xs = np.linspace(1, 0, n_frames_with_attack)
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
        xs = np.linspace(0, 1, n_frames_with_decay)
        decay = 1 - (1 - sustain_level) * xs ** decay_degree
    else:
        decay = np.array([])
    remaining_duration_in_frames -= len(decay)

    n_frames_with_sustain = min(
        remaining_duration_in_frames,
        floor(max_sustain_duration * frame_rate)
    )
    if n_frames_with_sustain > 0:
        sustain = sustain_level * np.ones(n_frames_with_sustain)
    else:
        sustain = np.array([])

    release_ratio = event.velocity ** release_duration_on_velocity_order
    release_duration = release_ratio * max_release_duration
    n_frames_with_release = floor(release_duration * frame_rate)
    if n_frames_with_release > 0:
        xs = np.linspace(0, 1, n_frames_with_release)
        release = sustain_level * (1 - xs ** release_degree)
    else:
        release = np.array([])

    envelope = np.concatenate((attack, hold, decay, sustain, release))
    envelope *= peak_value
    coef = event.velocity ** envelope_values_on_velocity_order
    envelope *= ratio_at_zero_velocity + coef * (1 - ratio_at_zero_velocity)
    return envelope


def create_relative_ahdsr_envelope(
        event: 'sinethesizer.synth.core.Event',
        attack_to_ahds_ratio: float = 0.2,
        attack_degree: float = 1.0,
        hold_to_ahds_ratio: float = 0.05,
        decay_to_ahds_ratio: float = 0.25,
        decay_degree: float = 1.0,
        sustain_level: float = 0.6,
        max_release_duration: float = 0.5,
        release_duration_on_velocity_order: float = 0.0,
        release_degree: float = 1.0,
        peak_value: float = 1.0,
        ratio_at_zero_velocity: float = 0.0,
        envelope_values_on_velocity_order: float = 0.0
) -> np.ndarray:
    """
    Create AHDSR envelope with proportional durations of stages.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and velocity
    :param attack_to_ahds_ratio:
        fraction of frames with attack amongst frames with attack, hold,
        decay, and sustain
    :param attack_degree:
        degree of attack dynamic; if it is 1, attack is linear; if it is
        greater than 1, attack is concave; if it is less than 1,
        attack is convex
    :param hold_to_ahds_ratio:
        fraction of frames with hold amongst frames with attack, hold,
        decay, and sustain
    :param decay_to_ahds_ratio:
        fraction of frames with decay amongst frames with attack, hold,
        decay, and sustain
    :param decay_degree:
        degree of decay dynamic; if it is 1, decay is linear; if it is
        greater than 1, decay is concave; if it is less than 1,
        decay is convex
    :param sustain_level:
        amplitude level at sustain stage; a float between 0 and 1 where 1 is
        the peak level (i.e., level at the end of attack)
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
    ahds_duration_in_frames = ceil(event.duration * frame_rate)

    n_frames_with_attack = floor(
        attack_to_ahds_ratio * ahds_duration_in_frames
    )
    if n_frames_with_attack > 0:
        xs = np.linspace(1, 0, n_frames_with_attack)
        attack = 1 - xs ** attack_degree
    else:
        attack = np.array([])

    n_frames_with_hold = floor(hold_to_ahds_ratio * ahds_duration_in_frames)
    hold = np.ones(n_frames_with_hold)

    n_frames_with_decay = floor(
        decay_to_ahds_ratio * ahds_duration_in_frames
    )
    if n_frames_with_decay > 0:
        xs = np.linspace(0, 1, n_frames_with_decay)
        decay = 1 - (1 - sustain_level) * xs ** decay_degree
    else:
        decay = np.array([])

    n_frames_with_sustain = (
        ahds_duration_in_frames - len(attack) - len(hold) - len(decay)
    )
    if n_frames_with_sustain > 0:
        sustain = sustain_level * np.ones(n_frames_with_sustain)
    else:
        sustain = np.array([])

    release_ratio = event.velocity ** release_duration_on_velocity_order
    release_duration = release_ratio * max_release_duration
    n_frames_with_release = floor(release_duration * frame_rate)
    if n_frames_with_release > 0:
        xs = np.linspace(0, 1, n_frames_with_release)
        release = sustain_level * (1 - xs ** release_degree)
    else:
        release = np.array([])

    envelope = np.concatenate((attack, hold, decay, sustain, release))
    envelope *= peak_value
    coef = event.velocity ** envelope_values_on_velocity_order
    envelope *= ratio_at_zero_velocity + coef * (1 - ratio_at_zero_velocity)
    return envelope


def create_trapezoid_envelope(
        event: 'sinethesizer.synth.core.Event',
        attack_share: float = 0.2,
        attack_degree: float = 1.0,
        decay_share: float = 0.1,
        decay_degree: float = 1.0,
        peak_value: float = 1.0,
        ratio_at_zero_velocity: float = 0.0,
        envelope_values_on_velocity_order: float = 0.0
) -> np.ndarray:
    """
    Create AHD envelope (so called trapezoid).

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and velocity
    :param attack_share:
        fraction of frames with attack amongst all frames
    :param attack_degree:
        degree of attack dynamic; if it is 1, attack is linear; if it is
        greater than 1, attack is concave; if it is less than 1,
        attack is convex
    :param decay_share:
        fraction of frames with decay amongst all frames
    :param decay_degree:
        degree of decay dynamic; if it is 1, decay is linear; if it is
        greater than 1, decay is concave; if it is less than 1,
        decay is convex
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
    envelope = create_relative_ahdsr_envelope(
        event,
        attack_to_ahds_ratio=attack_share,
        attack_degree=attack_degree,
        hold_to_ahds_ratio=max(1 - attack_share - decay_share, 0),
        decay_to_ahds_ratio=decay_share,
        decay_degree=decay_degree,
        sustain_level=0,
        max_release_duration=0,
        peak_value=peak_value,
        ratio_at_zero_velocity=ratio_at_zero_velocity,
        envelope_values_on_velocity_order=envelope_values_on_velocity_order
    )
    return envelope
