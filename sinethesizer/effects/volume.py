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
        new value of specified quantile of absolute pressure deviations
        at maximum velocity
    :param quantile:
        quantile of absolute pressure deviations that is used for scaling
    :param value_on_velocity_order:
        coefficient that determines dependence of amplitude quantile value
        on velocity
    :param value_at_zero_velocity:
        new value of specified quantile of absolute pressure deviations
        at zero velocity
    :return:
        sound with amplitude normalized to new value
    """
    coef = event.velocity ** value_on_velocity_order
    diff = value_at_max_velocity - value_at_zero_velocity
    new_quantile_value = value_at_zero_velocity + coef * diff
    quantile_value = np.quantile(np.abs(sound), quantile)
    sound *= new_quantile_value / quantile_value
    return sound
