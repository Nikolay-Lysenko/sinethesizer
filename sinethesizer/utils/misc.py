"""
Provide miscellaneous helper functions.

Author: Nikolay Lysenko
"""


import functools

import numpy as np


def mix_with_original_sound(fn):
    """
    Add support of argument named `original_sound_weight`.

    If this argument is passed, a signal processed by an effect is mixed
    with original sound.

    :param fn:
        effect function to be decorated; it must have `**kwargs` in its signature
    :return:
        function that depends on argument named `original_sound_weight`
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'original_sound_weight' not in kwargs:
            return fn(*args, **kwargs)
        original_sound_weight = kwargs.pop('original_sound_weight')
        if not 0 <= original_sound_weight <= 1:
            raise ValueError("Weight of original sound must be between 0 and 1.")
        original_sound = args[0] if args else kwargs['sound']
        processed_sound = fn(*args, **kwargs)
        output_sound = (
            original_sound_weight * original_sound
            + (1 - original_sound_weight) * processed_sound
        )
        return output_sound
    return wrapper


def sum_two_sounds(first_sound: np.ndarray, second_sound: np.ndarray) -> np.ndarray:
    """
    Sum two sounds of probably unequal durations.

    :param first_sound:
        first sound as array of shape (n_channels, n_frames)
    :param second_sound:
        second sound as array of shape (n_channels, n_frames)
    :return:
        sum of the sounds
    """
    first_n_frames = first_sound.shape[1]
    second_n_frames = second_sound.shape[1]
    n_extra_frames = abs(first_n_frames - second_n_frames)
    padding = np.zeros((first_sound.shape[0], n_extra_frames))
    if first_n_frames > second_n_frames:
        second_sound = np.hstack((second_sound, padding))
    elif first_n_frames < second_n_frames:
        first_sound = np.hstack((first_sound, padding))
    return first_sound + second_sound
