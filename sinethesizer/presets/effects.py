"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.synth.waves import generate_wave


def overdrive(
        sound: np.ndarray, frame_rate: int,
        fraction_to_clip: float = 0.1, strength: float = 0.3
) -> np.ndarray:
    """
    Overdrive the sound.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param fraction_to_clip:
        fraction of the most outlying frames to be hard clipped
    :param strength:
        relative strength of distortion, must be between 0 and 1
    :return:
        overdriven sound
    """
    if not (0 < fraction_to_clip < 1):
        raise ValueError("Fraction to clip must be between 0 and 1.")
    if not (0 < strength < 1):
        raise ValueError("Overdrive strength must be between 0 and 1.")
    _ = frame_rate  # All effects must have `frame_rate` argument.

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


def tremolo(
        sound: np.ndarray, frame_rate: int,
        frequency: float = 50, amplitude: float = 0.5
) -> np.ndarray:
    """
    Make sound volume vibrating.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param frequency:
        frequency of volume oscillations (in Hz)
    :param amplitude:
        relative amplitude of volume oscillation, must be between 0 and 1
    :return:
        sound with vibrating volume
    """
    if amplitude <= 0 or amplitude > 1:
        raise ValueError("Amplitude for tremolo must be between 0 and 1.")
    amplitudes = amplitude * np.ones(sound.shape[1])
    volume_wave = generate_wave(
        'sine', frequency, amplitudes,
        location=0, max_channel_delay=0, frame_rate=frame_rate
    )
    volume_wave += 1
    sound *= volume_wave
    return sound
