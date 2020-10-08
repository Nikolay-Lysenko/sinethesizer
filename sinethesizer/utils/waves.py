"""
Generate basic periodic waves.

Author: Nikolay Lysenko
"""


from typing import Optional

from functools import partial

import numpy as np
import scipy.signal


NAME_TO_WAVEFORM = {
    'sine': np.sin,
    'square': scipy.signal.square,
    'triangle': partial(scipy.signal.sawtooth, width=0.5),
    'sawtooth': scipy.signal.sawtooth,
    'white_noise': lambda array: np.random.normal(0, 1, array.shape),
}


def generate_mono_wave(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float = 0,
        modulator: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate wave with exactly one channel.

    :param waveform:
        form of wave; it can be one of 'sine', 'square', 'triangle',
        'sawtooth', and 'white_noise'
    :param frequency:
        frequency of wave (in Hz)
    :param amplitude_envelope:
        amplitude envelope; it also defines duration of sound
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift (in radians)
    :param modulator:
        wave that modulates frequency
    :return:
        sound wave as array of shape (1, len(amplitude_envelope))
    """
    duration_in_frames = len(amplitude_envelope)
    time_moments_in_seconds = np.arange(duration_in_frames) / frame_rate
    wave_fn = NAME_TO_WAVEFORM[waveform]
    if modulator is None:
        wave = wave_fn(
            2 * np.pi * frequency * time_moments_in_seconds + phase
        )
    else:
        wave = wave_fn(
            2 * np.pi * frequency * time_moments_in_seconds + phase + modulator
        )
    wave *= amplitude_envelope
    return wave
