"""
Create common interface for various periodic and non-periodic wave generators.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import Optional

import numpy as np
import scipy.signal

from sinethesizer.oscillators.analog import (
    generate_sawtooth_wave, generate_square_wave, generate_triangle_wave
)
from sinethesizer.oscillators.karplus_strong import (
    generate_karplus_strong_wave
)
from sinethesizer.oscillators.noise import generate_power_law_noise


TWO_PI = 2 * np.pi
PLAIN_ANALOG_WAVEFORMS = ['sine', 'raw_sawtooth', 'raw_square', 'raw_triangle']
BANDLIMITED_ANALOG_WAVEFORMS = ['sawtooth', 'square', 'triangle']
ANALOG_WAVEFORMS = PLAIN_ANALOG_WAVEFORMS + BANDLIMITED_ANALOG_WAVEFORMS
NOISES = ['white_noise', 'pink_noise', 'brown_noise']
MODEL_BASED_WAVEFORMS = ['karplus_strong']


def generate_analog_wave(
        waveform: str, frequency: float, duration_in_frames: int,
        frame_rate: int, phase: float = 0,
        phase_modulator: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate wave from an analog synthesizer with constant amplitude envelope.

    :param waveform:
        form of wave; it can be one of 'sine', 'sawtooth', 'square',
        'triangle', 'raw_sawtooth', 'raw_square', and 'raw_triangle'
    :param frequency:
        frequency of wave (in Hz)
    :param duration_in_frames:
        duration of output sound in frames
    :param frame_rate:
        number of frames per second
    :param phase:
        phase shift (in radians)
    :param phase_modulator:
        modulator for PM (phase modulation)
    :return:
        wave with constant amplitude envelope
    """
    name_to_waveform = {
        'sine': np.sin,
        'sawtooth': generate_sawtooth_wave,
        'square': generate_square_wave,
        'triangle': generate_triangle_wave,
        'raw_sawtooth': scipy.signal.sawtooth,
        'raw_square': scipy.signal.square,
        'raw_triangle': partial(scipy.signal.sawtooth, width=0.5),
    }
    wave_fn = name_to_waveform[waveform]

    moments_in_seconds = np.arange(duration_in_frames) / frame_rate
    if phase_modulator is None:
        xs = TWO_PI * frequency * moments_in_seconds + phase
    else:
        xs = TWO_PI * frequency * moments_in_seconds + phase + phase_modulator
    if waveform in PLAIN_ANALOG_WAVEFORMS:
        return wave_fn(xs)
    else:
        xs_step = TWO_PI * frequency / frame_rate
        return wave_fn(xs, xs_step)


def generate_model_based_waveform(
        waveform: str, frequency: float, duration_in_frames: int,
        frame_rate: int
) -> np.ndarray:
    """
    Generate wave with constant amplitude envelope based on a simulation model.

    :param waveform:
        form of wave; only 'karplus_strong' is supported now
    :param frequency:
        frequency of wave (in Hz)
    :param duration_in_frames:
        duration of output sound in frames
    :param frame_rate:
        number of frames per second
    :return:
        wave with constant amplitude envelope
    """
    name_to_waveform = {
        'karplus_strong': generate_karplus_strong_wave,
    }
    wave_fn = name_to_waveform[waveform]
    wave = wave_fn(frequency, duration_in_frames, frame_rate)
    return wave


def generate_noise(
        waveform: str, duration_in_frames: int, frame_rate: int
) -> np.ndarray:
    """
    Generate noise with constant amplitude envelope.

    :param waveform:
        form of wave; it can be one of 'white_noise', 'pink_noise',
        and 'brown_noise'
    :param duration_in_frames:
        duration of output sound in frames
    :param frame_rate:
        number of frames per second
    :return:
        noise with constant amplitude envelope
    """
    name_to_waveform = {
        'white_noise': lambda n_frames: np.random.normal(0, 0.3, n_frames),
        'pink_noise': partial(generate_power_law_noise, psd_decay_order=1),
        'brown_noise': partial(generate_power_law_noise, psd_decay_order=2),
    }
    wave_fn = name_to_waveform[waveform]
    if waveform == 'white_noise':
        return wave_fn(duration_in_frames)
    else:
        return wave_fn(duration_in_frames, frame_rate)


def generate_mono_wave(
        waveform: str, frequency: float, amplitude_envelope: np.ndarray,
        frame_rate: int, phase: float = 0,
        amplitude_modulator: Optional[np.ndarray] = None,
        phase_modulator: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate wave with exactly one channel.

    :param waveform:
        form of wave; it can be one of 'sine', 'sawtooth', 'square',
        'triangle', 'raw_sawtooth', 'raw_square', 'raw_triangle',
        'white_noise', 'pink_noise', 'brown_noise', and 'karplus_strong'
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
    if waveform in ANALOG_WAVEFORMS:
        wave = generate_analog_wave(
            waveform, frequency, duration_in_frames, frame_rate,
            phase, phase_modulator
        )
    elif waveform in MODEL_BASED_WAVEFORMS:
        wave = generate_model_based_waveform(
            waveform, frequency, duration_in_frames, frame_rate
        )
    elif waveform in NOISES:
        wave = generate_noise(waveform, duration_in_frames, frame_rate)
    else:
        raise ValueError(f"Unknown waveform: {waveform}.")

    wave *= amplitude_envelope
    if amplitude_modulator is not None:
        wave *= amplitude_modulator
    return wave
