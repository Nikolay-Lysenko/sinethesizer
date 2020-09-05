"""
Filter some frequencies from original sound.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import butter, sosfilt


def filter_absolute_frequencies(
        sound: np.ndarray, context: Dict[str, Any],
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = None,
        invert: bool = False, order: int = 10
) -> np.ndarray:
    """
    Filter some frequency ranges (defined in Hz) from original sound.

    :param sound:
        sound to be modified
    :param context:
        supplementary information about `sound`; it can contain number of
        frames per second and fundamental frequency (in Hz) of related event
    :param min_frequency:
        cutoff frequency for high-pass filtering (in Hz);
        there is no high-pass filtering by default
    :param max_frequency:
        cutoff frequency for low-pass filtering (in Hz);
        there is no low-pass filtering by default
    :param invert:
        if it is `True` and both `min_frequency` and `max_frequency`
        are passed, band-stop filter is applied instead of band-pass filter
    :param order:
        order of the filter; the higher it is, the steeper cutoff is
    :return:
        sound with some frequencies muted
    """
    invert = invert and min_frequency is not None and max_frequency is not None
    filter_type = 'bandstop' if invert else 'bandpass'
    nyquist_frequency = 0.5 * context['frame_rate']
    min_frequency = min_frequency or 1e-8  # Arbitrary small positive number.
    max_frequency = max_frequency or nyquist_frequency - 1e-8
    min_threshold = max(min_frequency / nyquist_frequency, 1e-8)
    max_threshold = min(max_frequency / nyquist_frequency, 1 - 1e-8)
    second_order_sections = butter(
        order, [min_threshold, max_threshold], btype=filter_type, output='sos'
    )  # 'ba' is not used, because sometimes it lacks numerical stability.
    sound = sosfilt(second_order_sections, sound)
    return sound


def filter_relative_frequencies(
        sound: np.ndarray, context: Dict[str, Any],
        min_frequency_ratio: Optional[float] = None,
        max_frequency_ratio: Optional[float] = None,
        invert: bool = False, order: int = 10
) -> np.ndarray:
    """
    Filter some frequency ranges (defined as ratios) from original sound.

    :param sound:
        sound to be modified
    :param context:
        supplementary information about `sound`; it can contain number of
        frames per second and fundamental frequency (in Hz) of related event
    :param min_frequency_ratio:
        ratio of cutoff frequency for high-pass filtering to fundamental
        frequency of the sound; there is no high-pass filtering by default
    :param max_frequency_ratio:
        ratio of cutoff frequency for low-pass filtering to fundamental
        frequency of the sound; there is no low-pass filtering by default
    :param invert:
        if it is `True` and both `min_frequency_ratio` and
        `max_frequency_ratio` are passed, band-stop filter is applied
        instead of band-pass filter
    :param order:
        order of the filter; the higher it is, the steeper cutoff is
    :return:
        sound with some frequencies muted
    """
    fundamental_frequency = context['fundamental_frequency']
    min_frequency = None
    if min_frequency_ratio is not None:
        min_frequency = min_frequency_ratio * fundamental_frequency
    max_frequency = None
    if max_frequency_ratio is not None:
        max_frequency = max_frequency_ratio * fundamental_frequency
    sound = filter_absolute_frequencies(
        sound, context, min_frequency, max_frequency, invert, order
    )
    return sound


def apply_frequency_filter(
        sound: np.ndarray, context: Dict[str, Any], kind: str = 'absolute',
        *args, **kwargs
) -> np.ndarray:
    """
    Filter some frequencies from original sound.

    :param sound:
        sound to be modified
    :param context:
        supplementary information about `sound`; it can contain number of
        frames per second and fundamental frequency (in Hz) of related event
    :param kind:
        kind of filter; supported values are 'absolute' and 'relative'
    :return:
        sound with some frequencies muted
    """
    if kind == 'absolute':
        sound = filter_absolute_frequencies(sound, context, *args, **kwargs)
    elif kind == 'relative':
        sound = filter_relative_frequencies(sound, context, *args, **kwargs)
    else:
        raise ValueError(
            f"Kind must be either 'absolute' or 'relative', but found: {kind}"
        )
    return sound
