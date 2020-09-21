"""
Filter some frequencies from original sound.

Note that only high-pass filter, low-pass filter, band-pass filter, and
band-stop filter are implemented in this module. Filters that can assign
arbitrary gain to an arbitrary frequency are called equalizers in this package,
so look at `equalizer` module for them.

Author: Nikolay Lysenko
"""


from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt


def filter_absolute_frequencies(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = None,
        invert: bool = False, order: int = 25
) -> np.ndarray:
    """
    Filter some frequency ranges (defined in Hz) from original sound.

    :param sound:
        sound to be modified
    :param event:
       parameters of sound event for which this function is called
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
    nyquist_frequency = 0.5 * event.frame_rate
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
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        min_frequency_ratio: Optional[float] = None,
        max_frequency_ratio: Optional[float] = None,
        invert: bool = False, order: int = 25
) -> np.ndarray:
    """
    Filter some frequency ranges (defined as ratios) from original sound.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
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
    fundamental_frequency = event.frequency
    min_frequency = None
    if min_frequency_ratio is not None:
        min_frequency = min_frequency_ratio * fundamental_frequency
    max_frequency = None
    if max_frequency_ratio is not None:
        max_frequency = max_frequency_ratio * fundamental_frequency
    sound = filter_absolute_frequencies(
        sound, event, min_frequency, max_frequency, invert, order
    )
    return sound


def filter_absolute_frequencies_wrt_velocity(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        min_frequency_at_zero_velocity: Optional[float] = None,
        min_frequency_at_max_velocity: Optional[float] = None,
        min_frequency_on_velocity_order: Optional[float] = None,
        max_frequency_at_zero_velocity: Optional[float] = None,
        max_frequency_at_max_velocity: Optional[float] = None,
        max_frequency_on_velocity_order: Optional[float] = None,
        invert: bool = False, order: int = 25
) -> np.ndarray:
    """
    Filter some frequencies (in Hz) depending on velocity.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param min_frequency_at_zero_velocity:
        cutoff frequency for high-pass filtering (in Hz) at zero velocity;
        there is no high-pass filtering by default
    :param min_frequency_at_max_velocity:
        cutoff frequency for high-pass filtering (in Hz) at maximum velocity;
        there is no high-pass filtering by default
    :param min_frequency_on_velocity_order:
        coefficient that defines dependence of cutoff frequency for high-pass
        filtering on velocity
    :param max_frequency_at_zero_velocity:
        cutoff frequency for low-pass filtering (in Hz) at zero velocity;
        there is no low-pass filtering by default
    :param max_frequency_at_max_velocity:
        cutoff frequency for low-pass filtering (in Hz) at maximum velocity;
        there is no low-pass filtering by default
    :param max_frequency_on_velocity_order:
        coefficient that defines dependence of cutoff frequency for low-pass
        filtering on velocity
    :param invert:
        if it is `True` and all preceding arguments are passed,
        band-stop filter is applied instead of band-pass filter
    :param order:
        order of the filter; the higher it is, the steeper cutoff is
    :return:
        sound with some frequencies muted
    """
    min_frequency = None
    if min_frequency_at_zero_velocity is not None:
        coef = event.velocity ** min_frequency_on_velocity_order
        min_frequency = (
            min_frequency_at_zero_velocity
            + coef * (
                min_frequency_at_max_velocity - min_frequency_at_zero_velocity
            )
        )
    max_frequency = None
    if max_frequency_at_zero_velocity is not None:
        coef = event.velocity ** max_frequency_on_velocity_order
        max_frequency = (
            max_frequency_at_zero_velocity
            + coef * (
                max_frequency_at_max_velocity - max_frequency_at_zero_velocity
            )
        )
    sound = filter_absolute_frequencies(
        sound, event, min_frequency, max_frequency, invert, order
    )
    return sound


def filter_relative_frequencies_wrt_velocity(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        min_frequency_ratio_at_zero_velocity: Optional[float] = None,
        min_frequency_ratio_at_max_velocity: Optional[float] = None,
        min_frequency_ratio_on_velocity_order: Optional[float] = None,
        max_frequency_ratio_at_zero_velocity: Optional[float] = None,
        max_frequency_ratio_at_max_velocity: Optional[float] = None,
        max_frequency_ratio_on_velocity_order: Optional[float] = None,
        invert: bool = False, order: int = 25
) -> np.ndarray:
    """
    Filter some frequencies (defined as ratios) depending on velocity.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param min_frequency_ratio_at_zero_velocity:
        ratio of cutoff frequency for high-pass filtering to fundamental
        frequency of the sound at zero velocity;
        there is no high-pass filtering by default
    :param min_frequency_ratio_at_max_velocity:
        ratio of cutoff frequency for high-pass filtering to fundamental
        frequency of the sound at maximum velocity;
        there is no high-pass filtering by default
    :param min_frequency_ratio_on_velocity_order:
        coefficient that defines dependence of cutoff frequency ratio
        for high-pass filtering on velocity
    :param max_frequency_ratio_at_zero_velocity:
        ratio of cutoff frequency for low-pass filtering to fundamental
        frequency of the sound at zero velocity;
        there is no low-pass filtering by default
    :param max_frequency_ratio_at_max_velocity:
        ratio of cutoff frequency for low-pass filtering to fundamental
        frequency of the sound at maximum velocity;
        there is no low-pass filtering by default
    :param max_frequency_ratio_on_velocity_order:
        coefficient that defines dependence of cutoff frequency ratio
        for low-pass filtering on velocity
    :param invert:
        if it is `True` and all preceding arguments are passed,
        band-stop filter is applied instead of band-pass filter
    :param order:
        order of the filter; the higher it is, the steeper cutoff is
    :return:
        sound with some frequencies muted
    """
    min_frequency_ratio = None
    if min_frequency_ratio_at_zero_velocity is not None:
        coef = event.velocity ** min_frequency_ratio_on_velocity_order
        min_frequency_ratio = (
            min_frequency_ratio_at_zero_velocity
            + coef * (
                min_frequency_ratio_at_max_velocity
                - min_frequency_ratio_at_zero_velocity
            )
        )
    max_frequency_ratio = None
    if max_frequency_ratio_at_zero_velocity is not None:
        coef = event.velocity ** max_frequency_ratio_on_velocity_order
        max_frequency_ratio = (
            max_frequency_ratio_at_zero_velocity
            + coef * (
                max_frequency_ratio_at_max_velocity
                - max_frequency_ratio_at_zero_velocity
            )
        )
    sound = filter_relative_frequencies(
        sound, event, min_frequency_ratio, max_frequency_ratio, invert, order
    )
    return sound


def apply_frequency_filter(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        kind: str = 'absolute', *args, **kwargs
) -> np.ndarray:
    """
    Filter some frequencies from original sound.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param kind:
        kind of filter; supported values are 'absolute' and 'relative'
    :return:
        sound with some frequencies muted
    """
    if kind == 'absolute':
        sound = filter_absolute_frequencies(sound, event, *args, **kwargs)
    elif kind == 'relative':
        sound = filter_relative_frequencies(sound, event, *args, **kwargs)
    elif kind == 'absolute_wrt_velocity':
        sound = filter_absolute_frequencies_wrt_velocity(
            sound, event, *args, **kwargs
        )
    elif kind == 'relative_wrt_velocity':
        sound = filter_relative_frequencies_wrt_velocity(
            sound, event, *args, **kwargs
        )
    else:
        raise ValueError(
            "Supported kinds are 'absolute', 'relative', "
            "'absolute_wrt_velocity', and 'relative_wrt_velocity', "
            f"but found: {kind}"
        )
    return sound
