"""
Control amplitude of sound (and hence its volume).

Author: Nikolay Lysenko
"""


import numpy as np


def apply_amplitude_normalization(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        value_at_max_velocity: float, quantile: float = 1,
        value_on_velocity_order: float = 1, value_at_zero_velocity: float = 0
) -> np.ndarray:
    """
    Normalize amplitude of sound.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param value_at_max_velocity:
        new value of specified quantile of absolute pressure deviations at maximum velocity
    :param quantile:
        quantile of absolute pressure deviations that is used for scaling
    :param value_on_velocity_order:
        coefficient that determines dependence of amplitude quantile value on velocity
    :param value_at_zero_velocity:
        new value of specified quantile of absolute pressure deviations at zero velocity
    :return:
        sound with amplitude normalized to new value
    """
    coef = event.velocity ** value_on_velocity_order
    diff = value_at_max_velocity - value_at_zero_velocity
    new_quantile_value = value_at_zero_velocity + coef * diff
    quantile_value = np.quantile(np.abs(sound), quantile)
    sound *= new_quantile_value / quantile_value
    return sound


def apply_compressor(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        threshold: float, frame_size_in_cycles: float = 3
) -> np.ndarray:
    """
    Limit maximum amplitude of the sound.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param threshold:
        ratio of maximum output amplitude to maximum possible amplitude that is not clipped by
        playing devices
    :param frame_size_in_cycles:
        size of one window to be processed independently (in number of fundamental frequency's
        periods); the higher it is, the less probable artifacts are, but also the slower
        compressor reaction is
    :return:
        sound of limited amplitude
    """
    scaling_coefs = []
    previous_ratio = 1
    frame_size_in_samples = frame_size_in_cycles * event.frame_rate / event.frequency
    for chunk in np.array_split(sound, frame_size_in_samples, axis=1):
        current_max = np.max(np.abs(chunk))
        current_ratio = min(threshold / current_max, 1)
        current_scaling_coefs = np.linspace(previous_ratio, current_ratio, chunk.shape[1], False)
        scaling_coefs.append(current_scaling_coefs)
        previous_ratio = current_ratio
    scaling_coefs = np.hstack(scaling_coefs)
    sound *= scaling_coefs
    return sound
