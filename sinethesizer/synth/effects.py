"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict, Optional

import numpy as np
from scipy.signal import butter, sosfilt

from sinethesizer.synth.waves import generate_wave


EFFECT_FN_TYPE = Callable[[np.ndarray, int],  np.ndarray]


def frequency_filter(
        sound: np.ndarray, frame_rate: int,
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = None,
        invert: bool = True, order: int = 10
) -> np.ndarray:
    """
    Filter some frequencies from original sound.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param min_frequency:
        cutoff frequency for high-pass filtering;
        there is no high-pass filtering by default
    :param max_frequency:
        cutoff frequency for low-pass filtering;
        there is no low-pass filtering by default
    :param invert:
        if it is `True` and both `min_frequency` and `max_frequency`
        are passed, band-stop filter is applied instead of band-pass filter
    :param order:
        order of the filter; the higher it is, the steeper cutoff is
    :return:
        sound with some frequencies muted
    """
    if invert and min_frequency is not None and max_frequency is not None:
        filter_type = 'bandstop'
    else:
        filter_type = 'bandpass'
    nyquist_frequency = 0.5 * frame_rate
    min_frequency = min_frequency or 1e-2  # Arbitrary small positive number.
    max_frequency = max_frequency or nyquist_frequency - 1e-2
    min_threshold = min_frequency / nyquist_frequency
    max_threshold = max_frequency / nyquist_frequency
    second_order_sections = butter(
        order, [min_threshold, max_threshold], btype=filter_type, output='sos'
    )  # 'ba' is not used, because sometimes it lacks numerical stability.
    sound = sosfilt(second_order_sections, sound)
    return sound


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
    if not (0 <= strength < 1):
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
        frequency: float = 6, amplitude: float = 0.5, waveform: str = 'sine'
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
        relative amplitude of volume oscillations, must be between 0 and 1
    :param waveform:
        form of volume oscillations wave
    :return:
        sound with vibrating volume
    """
    if not (0 < amplitude <= 1):
        raise ValueError("Amplitude for tremolo must be between 0 and 1.")
    amplitudes = amplitude * np.ones(sound.shape[1])
    volume_wave = generate_wave(waveform, frequency, amplitudes, frame_rate)
    volume_wave += 1
    sound *= volume_wave
    return sound


def vibrato(
        sound: np.ndarray, frame_rate: int,
        frequency: float = 4, width: float = 0.2, waveform: str = 'sine'
) -> np.ndarray:
    """
    Make sound frequency vibrating.

    :param sound:
        sound to be modified
    :param frame_rate:
        number of frames per second
    :param frequency:
        frequency of sound's frequency oscillations (in Hz)
    :param width:
        difference between the highest frequency of oscillating sound
        and the lowest frequency of oscillating sound (in semitones)
    :param waveform:
        form of frequency oscillations wave
    :return:
        sound with vibrating frequency
    """
    semitone = 2 ** (1 / 12)
    highest_to_lowest_ratio = semitone ** width
    # If x = 0, d(x + m * sin(2 * \pi * f * x))/dx = 1 + 2 * \pi * f * m.
    # If x = \pi, d(x + m * sin(2 * \pi * f * x))/dx = 1 - 2 * \pi * f * m.
    # Ratio of above right sides is `highest_to_lowest_ratio`.
    # Let us solve it for `m` (`max_delay`).
    max_delay = (
        (highest_to_lowest_ratio - 1)
        / ((highest_to_lowest_ratio + 1) * 2 * np.pi * frequency)
    )

    amplitudes = max_delay * frame_rate * np.ones(sound.shape[1])
    frequency_wave = generate_wave(waveform, frequency, amplitudes, frame_rate)
    time_indices = np.ones(sound.shape[1]).cumsum() - 1 + frequency_wave[0, :]

    upper_indices = np.ceil(time_indices).astype(int)
    upper_indices = np.clip(upper_indices, 0, sound.shape[1] - 1)
    upper_sound = sound[:, upper_indices]

    lower_indices = np.floor(time_indices).astype(int)
    lower_indices = np.clip(lower_indices, 0, sound.shape[1] - 1)
    lower_sound = sound[:, lower_indices]

    weights = time_indices - lower_indices
    sound = weights * upper_sound + (1 - weights) * lower_sound
    return sound


def get_effects_registry() -> Dict[str, EFFECT_FN_TYPE]:
    """
    Get mapping from effect names to functions that apply effects.

    :return:
        registry of effects
    """
    registry = {
        'filter': frequency_filter,
        'overdrive': overdrive,
        'tremolo': tremolo,
        'vibrato': vibrato
    }
    return registry
