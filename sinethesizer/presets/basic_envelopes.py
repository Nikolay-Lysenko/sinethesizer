"""
Define some simple ADSR (Attack-Decay-Sustain-Release) profiles.

Author: Nikolay Lysenko
"""


import numpy as np


def constant_with_linear_decrease(
        duration: int, decrease_share: float = 0.1
) -> np.ndarray:
    """
    Create envelope with amplitude that is constant everywhere except the end.

    :param duration:
        duration of sound in frames
    :param decrease_share:
        share of frames where amplitude linearly decreases from 1 to 0
    :return:
        envelope
    """
    n_frames_with_decrease = int(round(decrease_share * duration))
    step = -1 / n_frames_with_decrease
    decrease_part = np.arange(1, 0, step)
    constant_part = np.ones(duration - len(decrease_part))
    envelope = np.concatenate((constant_part, decrease_part))
    return envelope
