"""
Change power and amplitude distribution across frequencies.

Note that functions from this module rely on `scipy.signal.firwin2` and so
actual output is not always equal to the desired output. That being said,
the module can not be used for tasks that require high precision (e.g., narrow
notch filter), but it can be used for tasks where general proximity is enough
(e.g., imitation of resonating body).

Author: Nikolay Lysenko
"""


from typing import List

import numpy as np
from scipy.signal import convolve, firwin2


def equalize_with_absolute_frequencies(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        breakpoint_frequencies: List[float], gains: List[float],
        **kwargs
) -> np.ndarray:
    """
    Change power and amplitude distribution across frequencies.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param breakpoint_frequencies:
        frequencies (in Hz) that correspond to breaks in frequency response
        of equalizer
    :param gains:
        relative gains at corresponding breakpoint frequencies; a gain at an
        intermediate frequency is linearly interpolated
    :return:
        sound with altered frequency balance
    """
    nyquist_frequency = 0.5 * event.frame_rate
    breakpoint_frequencies = [
        min(x / nyquist_frequency, 1) for x in breakpoint_frequencies
    ]
    gains = [x for x in gains]  # Copy it to prevent modifying original list.
    if breakpoint_frequencies[0] != 0:
        breakpoint_frequencies.insert(0, 0)
        gains.insert(0, gains[0])
    if breakpoint_frequencies[-1] != 1:
        breakpoint_frequencies.append(1)
        gains.append(gains[-1])
    # `fir_size` is odd, because else there are constraints on `gains`.
    fir_size = 2 * int(round(event.frame_rate / 100)) + 1
    fir = firwin2(fir_size, breakpoint_frequencies, gains, **kwargs)
    sound = np.vstack((
        convolve(sound[0, :], fir, mode='same'),
        convolve(sound[1, :], fir, mode='same'),
    ))
    return sound


def equalize_with_relative_frequencies(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        breakpoint_frequencies_ratios: List[float], gains: List[float],
        **kwargs
) -> np.ndarray:
    """
    Change power and amplitude distribution across frequencies.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param breakpoint_frequencies_ratios:
        frequencies (represented as ratios to fundamental frequency) that
        correspond to breaks in frequency response of equalizer
    :param gains:
        relative gains at corresponding breakpoint frequencies; a gain at an
        intermediate frequency is linearly interpolated
    :return:
        sound with altered frequency balance
    """
    fundamental_frequency = event.frequency
    breakpoint_frequencies = [
        x * fundamental_frequency for x in breakpoint_frequencies_ratios
    ]
    sound = equalize_with_absolute_frequencies(
        sound, event, breakpoint_frequencies, gains, **kwargs
    )
    return sound


def apply_equalizer(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        kind: str = 'absolute', *args, **kwargs
) -> np.ndarray:
    """
    Change power and amplitude distribution across frequencies.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param kind:
        kind of filter; supported values are 'absolute' and 'relative'
    :return:
        sound with altered frequency balance
    """
    if kind == 'absolute':
        sound = equalize_with_absolute_frequencies(
            sound, event, *args, **kwargs
        )
    elif kind == 'relative':
        sound = equalize_with_relative_frequencies(
            sound, event, *args, **kwargs
        )
    else:
        raise ValueError(
            f"Supported kinds are 'absolute' and 'relative', but found: {kind}"
        )
    return sound
