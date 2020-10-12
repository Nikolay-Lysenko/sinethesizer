"""
Define additional envelopes.

Author: Nikolay Lysenko
"""


from math import ceil

import numpy as np


def create_constant_envelope(
        event: 'sinethesizer.synth.core.Event', value: float
) -> np.ndarray:
    """
    Create constant envelope.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and velocity
    :param value:
        value of the envelope
    :return:
        constant envelope
    """
    return value * np.ones(ceil(event.duration * event.frame_rate))


def create_proportional_to_frequency_constant_envelope(
        event: 'sinethesizer.synth.core.Event', ratio: float
) -> np.ndarray:
    """
    Create constant envelope with value linearly depending on frequency.

    This envelope can be used for adding vibrato with frequency or phase
    modulation.

    :param event:
        parameters of sound event for which this function is called;
        this argument provides information about duration, frame rate,
        and frequency
    :param ratio:
        ratio of envelope value to frequency (in Hz)
    :return:
        frequency-dependent constant envelope
    """
    value = ratio * event.frequency
    return value * np.ones(ceil(event.duration * event.frame_rate))
