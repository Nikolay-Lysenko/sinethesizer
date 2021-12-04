"""
Make left and right channels different.

Also look at `reverb.apply_room_reverb` function, since it is designed for sound spatialization.

Author: Nikolay Lysenko
"""


from math import ceil

import numpy as np


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


def apply_stereo_delay(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event', delay: float
) -> np.ndarray:
    """
    Delay one channel with respect to the other.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param delay:
        delay between channels (in seconds);
        if it is positive, left channel is delayed, if it is negative, right channel is delayed
    :return:
        sound with delay between channels
    """
    silence = np.zeros(ceil(abs(delay) * event.frame_rate))
    if delay >= 0:
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
