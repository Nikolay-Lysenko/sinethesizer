"""
Define functions describing dependence of partial's amplitude on task.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict, List


PARTIALS_AMPLITUDE_FN_TYPE = Callable[
    [int, 'sinethesizer.synth.core.Task'],
    float
]


def get_exponentially_decaying_amplitude(
        partial_id: int, task: 'sinethesizer.synth.core.Task',
        max_amplitudes: List[float], decay_degrees: List[float]
) -> float:
    """
    Get relative amplitude that exponentially decays with decreasing velocity.

    :param partial_id:
        position of partial in specifications of an instrument
    :param task:
        parameters of sound synthesis task that triggered generation of
        the partial
    :param max_amplitudes:
        all partials' amplitudes at maximum velocity; absolute values affect
        nothing, only their ratios matter
    :param decay_degrees:
        degrees of exponential decay for all partials
    :return:
        relative amplitude
    """
    max_amplitude = max_amplitudes[partial_id]
    velocity_correction = task.velocity ** decay_degrees[partial_id]
    amplitude = max_amplitude * velocity_correction
    return amplitude


def get_partials_amplitude_functions_registry() -> Dict[str, PARTIALS_AMPLITUDE_FN_TYPE]:
    """
    Get mapping from partials amplitude function names to the functions itself.

    :return:
        registry of partials amplitude functions
    """
    registry = {'exponentially_decaying': get_exponentially_decaying_amplitude}
    return registry
