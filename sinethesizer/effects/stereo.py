"""
Make a difference between left and right channels.

Author: Nikolay Lysenko
"""


from math import ceil

import numpy as np


def decrease_channel_volume(
        sound: np.ndarray, task: 'sinethesizer.synth.core.Task',
        channel: str, volume_ratio: float
) -> np.ndarray:
    """
    Decrease volume of one channel.

    :param sound:
        sound to be modified
    :param task:
        an argument that is not used by this function;
        it is added, because all effect functions must have it
    :param channel:
        channel to be decreased
    :param volume_ratio:
        ratio of new volume of channel to its initial volume
    :return:
        sound with changed volume
    """
    _ = task  # This argument is ignored.
    if channel == 'left':
        ratios = np.array([[volume_ratio], [1]])
    elif channel == 'right':
        ratios = np.array([[1], [volume_ratio]])
    else:
        raise ValueError(f"Unknown channel: {channel}.")
    sound *= ratios
    return sound


def apply_haas_effect(
        sound: np.ndarray, task: 'sinethesizer.synth.core.Task',
        location: float, max_channel_delay: float
) -> np.ndarray:
    """
    Apply Haas effect.

    :param sound:
        sound to be modified
    :param task:
        parameters of sound synthesis task that triggered this effect
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is correlated with size of imaginary space occupied by sound sources
    :return:
        sound with delay between channels
    """
    delay = max_channel_delay * abs(location)
    silence = np.zeros(ceil(delay * task.frame_rate))
    if location >= 0:
        result = np.vstack((
            np.hstack((silence, sound[0])),
            np.hstack((sound[1], silence))
        ))
    else:
        result = np.vstack((
            np.hstack((sound[0], silence)),
            np.hstack((silence, sound[1]))
        ))
    return result
