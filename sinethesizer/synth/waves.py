"""
Generate basic sound waves.

Author: Nikolay Lysenko
"""


from functools import partial
from math import ceil

import numpy as np
import scipy.signal


NAME_TO_WAVEFORM = {
    'sine': np.sin,
    'square': scipy.signal.square,
    'triangle': partial(scipy.signal.sawtooth, width=0.5),
    'sawtooth': scipy.signal.sawtooth
}


def generate_wave(
        form: str, frequency: float, amplitudes: np.ndarray,
        location: float, max_channel_delay: float, frame_rate: int,
        phase: int = 0,
) -> np.ndarray:
    """
    Generate sound wave.

    :param form:
        form of wave;
        it can be one of 'sine', 'square', 'triangle', and 'sawtooth'
    :param frequency:
        frequency of sine wave; it defines pitch of sound
    :param amplitudes:
        array of amplitudes for each time frame;
        it defines volume of sound, its duration, and its volume envelope
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is correlated with size of imaginary space occupied by sound sources
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift in frames
    :return:
        sound wave represented as timeline of pressure deviations
    """
    duration_in_frames = len(amplitudes)
    xs = np.arange(duration_in_frames) + phase
    wave_fn = NAME_TO_WAVEFORM[form]
    plain_wave = wave_fn(2 * np.pi * frequency / frame_rate * xs)
    left_wave = (1 - location) * amplitudes * plain_wave
    right_wave = (location + 1) * amplitudes * plain_wave
    delay = max_channel_delay * abs(location)
    silence = np.zeros(ceil(delay * frame_rate))
    if location >= 0:
        result_wave = np.vstack((
            np.hstack((silence, left_wave)),
            np.hstack((right_wave, silence))
        ))
    else:
        result_wave = np.vstack((
            np.hstack((left_wave, silence)),
            np.hstack((silence, right_wave))
        ))
    return result_wave
