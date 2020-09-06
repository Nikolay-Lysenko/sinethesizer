"""
Define dependence of partial's amplitude on its ID and event parameters.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict, List


PARTIALS_AMPLITUDE_FN_TYPE = Callable[
    [int, 'sinethesizer.synth.core.Event'],
    float
]


def get_exponentially_dependent_amplitude(
        partial_id: int, event: 'sinethesizer.synth.core.Event',
        max_amplitudes: List[float], decay_degrees: List[float]
) -> float:
    """
    Get relative amplitude that exponentially depends on velocity.

    :param partial_id:
        position of partial in representation of an instrument
    :param event:
        parameters of event for which this function is called
    :param max_amplitudes:
        all partials' amplitudes at maximum velocity; absolute values affect
        nothing, only their ratios matter
    :param decay_degrees:
        degrees of exponential decay for all partials
    :return:
        relative amplitude
    """
    max_amplitude = max_amplitudes[partial_id]
    velocity_correction = event.velocity ** decay_degrees[partial_id]
    amplitude = max_amplitude * velocity_correction
    return amplitude


def get_partials_amplitude_functions_registry(
) -> Dict[str, PARTIALS_AMPLITUDE_FN_TYPE]:
    """
    Get mapping from partials amplitude function names to the functions itself.

    :return:
        registry of partials' amplitude functions
    """
    registry = {
        'exponentially_dependent': get_exponentially_dependent_amplitude
    }
    return registry
