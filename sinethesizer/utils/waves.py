"""
Generate basic sound waves.

Author: Nikolay Lysenko
"""


from functools import partial
from math import ceil, pi, sqrt, tan

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
        it defines volume of sound, its duration, and its volume envelope
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


def apply_haas_effect(
        left_channel: np.ndarray, right_channel: np.ndarray, frame_rate: int,
        location: float, max_channel_delay: float
) -> np.ndarray:
    """
    Apply Haas effect.

    :param left_channel:
        timeline of left channel
    :param right_channel:
        timeline of right channel
    :param frame_rate:
        number of frames per second
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is correlated with size of imaginary space occupied by sound sources
    :return:
        2D array with mixed channels
    """
    delay = max_channel_delay * abs(location)
    silence = np.zeros(ceil(delay * frame_rate))
    if location >= 0:
        result = np.vstack((
            np.hstack((silence, left_channel)),
            np.hstack((right_channel, silence))
        ))
    else:
        result = np.vstack((
            np.hstack((left_channel, silence)),
            np.hstack((silence, right_channel))
        ))
    return result


def generate_stereo_wave(
        form: str, frequency: float, amplitudes: np.ndarray, frame_rate: int,
        location: float = 0, max_channel_delay: float = 0, phase: int = 0
) -> np.ndarray:
    """
    Generate sound wave with two channels (left and right).

    :param form:
        form of wave;
        it can be one of 'sine', 'square', 'triangle', and 'sawtooth'
    :param frequency:
        frequency of wave; it defines pitch of sound
    :param amplitudes:
        array of amplitudes for each time frame;
        it defines volume of sound, its duration, and its volume envelope
    :param frame_rate:
        number of frames per second
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds (for Haas effect);
        it is correlated with size of imaginary space occupied by sound sources
    :param phase:
        phase shift in frames
    :return:
        sound wave represented as timeline of pressure deviations
    """
    mono_wave = generate_mono_wave(
        form, frequency, amplitudes, frame_rate, phase
    )
    # Set total power of left and right channels to original power:
    # left_amplitude ** 2 + right_amplitude ** 2 = 1.
    # Also assume that:
    # right_amplitude / left_amplitude = k ** tan(math.pi / 2 * location),
    # where k is a constant parameter (k = 1.1 below).
    if location == 1:
        left_amplitude = 0
        right_amplitude = 1
    elif location == -1:
        left_amplitude = 1
        right_amplitude = 0
    else:
        amplitudes_ratio = 1.1 ** tan(pi / 2 * location)
        left_amplitude = sqrt(1 / (1 + amplitudes_ratio**2))
        right_amplitude = sqrt(amplitudes_ratio**2 / (1 + amplitudes_ratio**2))
    left_wave = left_amplitude * mono_wave
    right_wave = right_amplitude * mono_wave
    result_wave = apply_haas_effect(
        left_wave, right_wave, frame_rate, location, max_channel_delay
    )
    return result_wave
