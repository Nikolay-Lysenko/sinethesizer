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


def generate_mono_wave(
        form: str, frequency: float, amplitudes: np.ndarray, frame_rate: int,
        phase: int = 0
) -> np.ndarray:
    """
    Generate wave with exactly one channel.

    :param form:
        form of wave;
        it can be one of 'sine', 'square', 'triangle', and 'sawtooth'
    :param frequency:
        frequency of wave; it defines pitch of sound
    :param amplitudes:
        array of amplitudes for each time frame;
        it defines duration of sound, its volume envelope and its max volume
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift in frames
    :return:
        sound wave as array of shape (1, len(amplitudes))
    """
    duration_in_frames = len(amplitudes)
    xs = np.arange(duration_in_frames) + phase
    wave_fn = NAME_TO_WAVEFORM[form]
    wave = amplitudes * wave_fn(2 * np.pi * frequency / frame_rate * xs)
    return wave
