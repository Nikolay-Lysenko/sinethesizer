"""
Define multiplicative contribution of event parameters to amplitude.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict


EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE = Callable[
    ['sinethesizer.synth.core.Event'],
    float
]


def compute_amplitude_factor_as_power_of_velocity(
        event: 'sinethesizer.synth.core.Event', power: float
) -> float:
    """
    Compute amplitude factor as a power function of velocity.

    :param event:
        parameters of event for which this function is called
    :param power:
        power of the function
    :return:
        multiplicative contribution of velocity to amplitude
    """
    amplitude_factor = event.velocity ** power
    return amplitude_factor


def get_event_to_amplitude_factor_functions_registry(
) -> Dict[str, EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE]:
    """
    Get mapping from amplitude factor functions' names to the functions itself.

    :return:
        registry of amplitude factor functions
    """
    registry = {
        'power_fn_of_velocity': compute_amplitude_factor_as_power_of_velocity,
    }
    return registry
