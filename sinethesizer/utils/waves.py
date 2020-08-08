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
        stereo_wave: np.ndarray, frame_rate: int, location: float,
        max_channel_delay: float
) -> np.ndarray:
    """
    Apply Haas effect.

    :param stereo_wave:
        2D array with left and right timeline
    :param frame_rate:
        number of frames per second
    :param location:
        location of sound source;
        -1 stands for extremely left and 1 stands for extremely right
    :param max_channel_delay:
        maximum possible delay between channels in seconds;
        it is correlated with size of imaginary space occupied by sound sources
    :return:
        2D array with delay between channels
    """
    delay = max_channel_delay * abs(location)
    silence = np.zeros(ceil(delay * frame_rate))
    if location >= 0:
        result = np.vstack((
            np.hstack((silence, stereo_wave[0])),
            np.hstack((stereo_wave[1], silence))
        ))
    else:
        result = np.vstack((
            np.hstack((stereo_wave[0], silence)),
            np.hstack((silence, stereo_wave[1]))
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
    left_amplitude = (1 - max(location, 0))
    right_amplitude = (1 + min(location, 0))
    left_wave = left_amplitude * mono_wave
    right_wave = right_amplitude * mono_wave
    stereo_wave = np.vstack((left_wave, right_wave))
    stereo_wave = apply_haas_effect(
        stereo_wave, frame_rate, location, max_channel_delay
    )
    return stereo_wave
