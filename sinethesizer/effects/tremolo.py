"""
Make sound volume vibrating.

Tremolo is also known as oscillator for amplitude.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.utils.waves import generate_mono_wave


def apply_absolute_tremolo(
        sound: np.ndarray, task: 'sinethesizer.synth.core.Task',
        frequency: float = 6, amplitude: float = 0.5,
        waveform: str = 'sine'
) -> np.ndarray:
    """
    Make sound volume vibrating with frequency defined in Hz.

    :param sound:
        sound to be modified
    :param task:
        parameters of sound synthesis task that triggered this effect
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
    volume_wave = generate_mono_wave(
        waveform, frequency, amplitudes, task.frame_rate
    )
    volume_wave += 1
    sound *= volume_wave
    return sound


def apply_relative_tremolo(
        sound: np.ndarray, task: 'sinethesizer.synth.core.Task',
        frequency_ratio: float = 0.02, amplitude: float = 0.5,
        waveform: str = 'sine'
) -> np.ndarray:
    """
    Make sound volume vibrating with frequency depending on that of the sound.

    :param sound:
        sound to be modified
    :param task:
        parameters of sound synthesis task that triggered this effect
    :param frequency_ratio:
        frequency of volume oscillations as ratio to fundamental frequency
        of the sound
    :param amplitude:
        relative amplitude of volume oscillations, must be between 0 and 1
    :param waveform:
        form of volume oscillations wave
    :return:
        sound with vibrating volume
    """
    frequency = frequency_ratio * task.frequency
    sound = apply_absolute_tremolo(sound, task, frequency, amplitude, waveform)
    return sound


def apply_tremolo(
        sound: np.ndarray, task: 'sinethesizer.synth.core.Task',
        kind: str = 'absolute', *args, **kwargs
) -> np.ndarray:
    """
    Make sound volume vibrating.

    :param sound:
        sound to be modified
    :param task:
        parameters of sound synthesis task that triggered this effect
    :param kind:
        kind of filter; supported values are 'absolute' and 'relative'
    :return:
        sound with vibrating volume
    """
    if kind == 'absolute':
        sound = apply_absolute_tremolo(sound, task, *args, **kwargs)
    elif kind == 'relative':
        sound = apply_relative_tremolo(sound, task, *args, **kwargs)
    else:
        raise ValueError(
            f"Kind must be either 'absolute' or 'relative', but found: {kind}"
        )
    return sound
