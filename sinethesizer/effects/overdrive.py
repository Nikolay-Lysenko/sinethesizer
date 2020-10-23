"""
Overdrive (distort) sound.

Author: Nikolay Lysenko
"""


import numpy as np


def apply_overdrive(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        fraction_to_clip: float = 0.1, strength: float = 0.333
) -> np.ndarray:
    """
    Overdrive the sound.

    :param sound:
        sound to be modified
    :param event:
        an argument that is not used by this function;
        it is added, because all effect functions must have it
    :param fraction_to_clip:
        fraction of the most outlying frames to be hard clipped
    :param strength:
        relative strength of distortion, must be between 0 and 1
    :return:
        overdriven sound
    """
    if not (0 < fraction_to_clip < 1):
        raise ValueError("Fraction to clip must be between 0 and 1.")
    if not (0 <= strength < 1):
        raise ValueError("Overdrive strength must be between 0 and 1.")
    _ = event  # This argument is ignored.

    abs_sound = np.abs(sound)
    clipping_threshold = np.quantile(abs_sound, 1 - fraction_to_clip, axis=1)
    clipping_threshold = clipping_threshold.reshape((-1, 1))
    clipping_cond = abs_sound >= clipping_threshold
    distorted_sound = sound - strength * sound**3 / clipping_threshold**2
    clipped_sound = np.sign(sound) * (1 - strength) * clipping_threshold
    sound = (
        ~clipping_cond * distorted_sound
        + clipping_cond * clipped_sound
    )
    sound /= (1 - strength)
    return sound
