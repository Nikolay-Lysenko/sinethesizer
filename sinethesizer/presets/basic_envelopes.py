"""
Define some simple ADSR (Attack-Decay-Sustain-Release) profiles.

Author: Nikolay Lysenko
"""


import numpy as np


def constant_with_linear_ends(
        duration: int, end_share: float = 0.1
) -> np.ndarray:
    """
    Create envelope with amplitude that is constant everywhere except its ends.

    :param duration:
        duration of sound in frames
    :param end_share:
        share of frames where amplitude changes linearly;
        actual number of frames with linear dynamic is two times higher,
        because there are left end with increase and right end with decrease
    :return:
        envelope
    """
    duration_of_linearity = int(round(end_share * duration))
    step = 1 / duration_of_linearity
    increase_part = np.arange(0, 1, step)
    decrease_part = np.arange(1, 0, -step)
    constant_part = np.ones(duration - len(increase_part) - len(decrease_part))
    envelope = np.concatenate((increase_part, constant_part, decrease_part))
    return envelope
