"""
Make sound frequency vibrating.

Vibrato is also known as oscillator for pitch.

Author: Nikolay Lysenko
"""


import numpy as np

from sinethesizer.oscillators import generate_mono_wave


def apply_absolute_vibrato(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        frequency: float = 4, width: float = 0.2,
        waveform: str = 'sine'
) -> np.ndarray:
    """
    Make sound frequency (i.e., pitch) vibrating with frequency defined in Hz.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
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

    amplitude_envelope = max_delay * event.frame_rate * np.ones(sound.shape[1])
    frequency_wave = generate_mono_wave(
        waveform, frequency, amplitude_envelope, event.frame_rate
    )
    time_indices = np.ones(sound.shape[1]).cumsum() - 1 + frequency_wave

    upper_indices = np.ceil(time_indices).astype(int)
    upper_indices = np.clip(upper_indices, 0, sound.shape[1] - 1)
    upper_sound = sound[:, upper_indices]

    lower_indices = np.floor(time_indices).astype(int)
    lower_indices = np.clip(lower_indices, 0, sound.shape[1] - 1)
    lower_sound = sound[:, lower_indices]

    weights = time_indices - lower_indices
    sound = weights * upper_sound + (1 - weights) * lower_sound
    return sound


def apply_relative_vibrato(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        frequency_ratio: float = 0.05, width: float = 0.2,
        waveform: str = 'sine'
) -> np.ndarray:
    """
    Make sound frequency vibrating with frequency depending on sound's pitch.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param frequency_ratio:
        frequency of sound's frequency oscillations as ratio
        to fundamental frequency
    :param width:
        difference between the highest frequency of oscillating sound
        and the lowest frequency of oscillating sound (in semitones)
    :param waveform:
        form of frequency oscillations wave
    :return:
        sound with vibrating frequency
    """
    frequency = frequency_ratio * event.frequency
    sound = apply_absolute_vibrato(sound, event, frequency, width, waveform)
    return sound


def apply_vibrato(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        kind: str = 'absolute', *args, **kwargs
) -> np.ndarray:
    """
    Make sound frequency (i.e., pitch) vibrating.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param kind:
        kind of filter; supported values are 'absolute' and 'relative'
    :return:
        sound with vibrating pitch
    """
    if kind == 'absolute':
        sound = apply_absolute_vibrato(sound, event, *args, **kwargs)
    elif kind == 'relative':
        sound = apply_relative_vibrato(sound, event, *args, **kwargs)
    else:
        raise ValueError(
            f"Kind must be either 'absolute' or 'relative', but found: {kind}"
        )
    return sound
