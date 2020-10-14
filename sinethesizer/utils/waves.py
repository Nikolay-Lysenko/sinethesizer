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
    'white_noise': partial(generate_power_law_noise, psd_decay_order=0),
    'pink_noise': partial(generate_power_law_noise, psd_decay_order=1),
    'brown_noise': partial(generate_power_law_noise, psd_decay_order=2),
}


def generate_mono_wave_without_amplitude_envelope(
        waveform: str, frequency: float, duration_in_frames: int,
        frame_rate: int, frequency_modulator: Optional[np.ndarray],
        phase_modulator: Optional[np.ndarray], phase: float = 0
) -> np.ndarray:
    """
    Generate wave with one channel and with trivial amplitude envelope.

    :param waveform:
        form of wave; it can be one of 'sine', 'square', 'triangle',
        'sawtooth', 'white_noise', 'pink_noise', and 'brown_noise'
    :param frequency:
        frequency of wave (in Hz)
    :param duration_in_frames:
        duration of sound in frames
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift (in radians)
    :param frequency_modulator:
        modulator for FM (frequency modulation)
    :param phase_modulator:
        modulator for PM (phase modulation)
    :return:
        wave with one channel and with trivial amplitude envelope
    """
    moments_in_seconds = np.arange(duration_in_frames) / frame_rate
    wave_fn = NAME_TO_WAVEFORM[waveform]
    args_dict = {'pink_noise': [frame_rate], 'brown_noise': [frame_rate]}
    args = args_dict.get(waveform, [])
    if frequency_modulator is None and phase_modulator is None:
        wave = wave_fn(
            2 * np.pi * frequency * moments_in_seconds
            + phase,
            *args
        )
    elif frequency_modulator is None:
        wave = wave_fn(
            2 * np.pi * frequency * moments_in_seconds
            + phase + phase_modulator,
            *args
        )
    elif phase_modulator is None:
        wave = wave_fn(
            2 * np.pi * (frequency + frequency_modulator) * moments_in_seconds
            + phase,
            *args
        )
    else:
        wave = wave_fn(
            2 * np.pi * (frequency + frequency_modulator) * moments_in_seconds
            + phase + phase_modulator,
            *args
        )
    return wave


def add_amplitude_envelope(
        wave: np.ndarray, amplitude_envelope: np.ndarray,
        amplitude_modulator: Optional[np.ndarray],
        ring_modulator: Optional[np.ndarray]
) -> np.ndarray:
    """
    Add amplitude envelope and amplitude/ring modulation.

    :param wave:
        wave to be modified
    :param amplitude_envelope:
        amplitude envelope
    :param amplitude_modulator:
        modulator for AM (amplitude modulation)
    :param ring_modulator:
        modulator for RM (ring modulation)
    :return:
        wave with non-trivial amplitude envelope and amplitude/ring modulation
    """
    wave *= amplitude_envelope
    if amplitude_modulator is not None:
        wave *= (1 + amplitude_modulator)
    if ring_modulator is not None:
        wave *= ring_modulator
    return wave


def generate_mono_wave(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float = 0,
        amplitude_modulator: Optional[np.ndarray] = None,
        ring_modulator: Optional[np.ndarray] = None,
        frequency_modulator: Optional[np.ndarray] = None,
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
        modulator for AM (amplitude modulation)
    :param ring_modulator:
        modulator for RM (ring modulation)
    :param frequency_modulator:
        modulator for FM (frequency modulation)
    :param phase_modulator:
        modulator for PM (phase modulation)
    :return:
        sound wave as array of shape (1, len(amplitude_envelope))
    """
    duration_in_frames = len(amplitude_envelope)
    wave = generate_mono_wave_without_amplitude_envelope(
        waveform, frequency, duration_in_frames, frame_rate,
        frequency_modulator, phase_modulator, phase
    )
    wave = add_amplitude_envelope(
        wave, amplitude_envelope, amplitude_modulator, ring_modulator
    )
    return wave
