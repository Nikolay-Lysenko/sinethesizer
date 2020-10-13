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
