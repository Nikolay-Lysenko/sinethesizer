"""
Generate basic sound waves.

Author: Nikolay Lysenko
"""


from functools import partial

import numpy as np
import scipy.signal


NAME_TO_WAVEFORM = {
    'sine': np.sin,
    'square': scipy.signal.square,
    'triangle': partial(scipy.signal.sawtooth, width=0.5),
    'sawtooth': scipy.signal.sawtooth
}


def generate_wave(
        form: str, frequency: float, frame_rate: int,
        amplitudes: np.ndarray, phase: int = 0
) -> np.ndarray:
    """
    Generate sound wave.

    :param form:
        form of wave;
        it can be one of 'sine', 'square', 'triangle', and 'sawtooth'
    :param frequency:
        frequency of sine wave; it defines pitch of sound
    :param frame_rate:
        number of frames per second
    :param amplitudes:
        array of amplitudes for each time frame;
        it defines volume of sound, its duration, and its ADSR envelope
    :param phase:
        phase shift in frames
    :return:
        sound wave represented as timeline of pressure deviations
    """
    duration_in_frames = len(amplitudes)
    xs = np.arange(duration_in_frames) + phase
    wave_fn = NAME_TO_WAVEFORM[form]
    plain_wave = wave_fn(2 * np.pi * frequency / frame_rate * xs)
    result_wave = amplitudes * plain_wave
    return result_wave
