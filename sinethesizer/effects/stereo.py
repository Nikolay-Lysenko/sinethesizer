"""
Make left and right channels different.

Author: Nikolay Lysenko
"""


from math import ceil

import numpy as np


def apply_haas_effect(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        location: float, max_channel_delay: float
) -> np.ndarray:
    """
    Apply Haas effect.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param location:
        location of sound source; a float between -1 and 1 where
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is correlated with size of imaginary space occupied by sound sources
    :return:
        sound with delay between channels
    """
    delay = max_channel_delay * abs(location)
    silence = np.zeros(ceil(delay * event.frame_rate))
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


def apply_panning(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        left_volume_ratio: float, right_volume_ratio: float
) -> np.ndarray:
    """
    Modify volumes of two channels independently.

    :param sound:
        sound to be modified
    :param event:
        an argument that is not used by this function;
        it is added, because all effect functions must have it
    :param left_volume_ratio:
        ratio of new volume of left channel to its initial volume
    :param right_volume_ratio:
        ratio of new volume of right channel to its initial volume
    :return:
        sound with changed volume
    """
    _ = event  # This argument is ignored.
    sound *= np.array([[left_volume_ratio], [right_volume_ratio]])
    return sound
