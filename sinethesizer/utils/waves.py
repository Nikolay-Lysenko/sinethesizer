"""
Generate basic waves with modulations.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Optional

import numpy as np
import scipy.signal

from sinethesizer.utils.noise import generate_power_law_noise


NAME_TO_WAVEFORM = {
    'sine': np.sin,
    'square': scipy.signal.square,
    'triangle': partial(scipy.signal.sawtooth, width=0.5),
    'sawtooth': scipy.signal.sawtooth,
    'white_noise': lambda xs: np.random.normal(0, 0.3, xs.shape),
    'pink_noise': partial(generate_power_law_noise, psd_decay_order=1),
    'brown_noise': partial(generate_power_law_noise, psd_decay_order=2),
}


def generate_mono_wave(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float = 0,
        amplitude_modulator: Optional[np.ndarray] = None,
        phase_modulator: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate wave with exactly one channel.

    :param waveform:
        form of wave; it can be one of 'sine', 'square', 'triangle',
        'sawtooth', 'white_noise', 'pink_noise', and 'brown_noise'
    :param frequency:
        frequency of wave (in Hz)
    :param amplitude_envelope:
        amplitude envelope; it also defines duration of sound
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift (in radians)
    :param amplitude_modulator:
        modulator for AM (amplitude modulation) or RM (ring modulation)
    :param phase_modulator:
        modulator for PM (phase modulation)
    :return:
        sound wave as array of shape (1, len(amplitude_envelope))
    """
    duration_in_frames = len(amplitude_envelope)
    moments_in_seconds = np.arange(duration_in_frames) / frame_rate
    wave_fn = NAME_TO_WAVEFORM[waveform]
    args_dict = {'pink_noise': [frame_rate], 'brown_noise': [frame_rate]}
    args = args_dict.get(waveform, [])
    if phase_modulator is None:
        wave = wave_fn(
            2 * np.pi * frequency * moments_in_seconds
            + phase,
            *args
        )
    else:
        wave = wave_fn(
            2 * np.pi * frequency * moments_in_seconds
            + phase + phase_modulator,
            *args
        )
    wave *= amplitude_envelope
    if amplitude_modulator is not None:
        wave *= amplitude_modulator
    return wave
