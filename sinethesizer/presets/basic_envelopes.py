"""
Define some simple ADSR (Attack-Decay-Sustain-Release) profiles.

Author: Nikolay Lysenko
"""


import numpy as np


def wide_spike(duration: int, breakpoint_share: float = 0.2) -> np.ndarray:
    """
    Create envelope with amplitude that grows linearly and then falls linearly.

    :param duration:
        duration of sound in frames
    :param breakpoint_share:
        share of frames that are before the breakpoint
    :return:
        envelope
    """
    duration_of_increase = int(round(breakpoint_share * duration))
    step = 1 / duration_of_increase
    increase_part = np.arange(0, 1, step)

    duration_of_decrease = duration - duration_of_increase
    step = 1 / duration_of_decrease
    decrease_part = np.arange(1, 0, -step)

    envelope = np.concatenate((increase_part, decrease_part))
    return envelope


def constant_with_linear_ends(
        duration: int, begin_share: float = 0.1, end_share: float = 0.1
) -> np.ndarray:
    """
    Create envelope with amplitude that is constant everywhere except its ends.

    :param duration:
        duration of sound in frames
    :param begin_share:
        share of first frames where amplitude increases linearly
    :param end_share:
        share of last frames where amplitude decreases linearly
    :return:
        envelope
    """
    duration_of_increase = int(round(begin_share * duration))
    step = 1 / duration_of_increase
    increase_part = np.arange(0, 1, step)

    duration_of_decrease = int(round(end_share * duration))
    step = 1 / duration_of_decrease
    decrease_part = np.arange(1, 0, -step)

    constant_part = np.ones(duration - len(increase_part) - len(decrease_part))
    envelope = np.concatenate((increase_part, constant_part, decrease_part))
    return envelope
