"""
Map names of envelopes to functions creating them.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict

import numpy as np

from sinethesizer.envelopes.ahdsr import (
    generic_ahdsr, relative_ahdsr, trapezoid
)
from sinethesizer.envelopes.misc import constant
from sinethesizer.envelopes.user_defined import user_defined_envelope


ENVELOPE_FN_TYPE = Callable[['sinethesizer.synth.core.Event'], np.ndarray]


def get_envelopes_registry() -> Dict[str, ENVELOPE_FN_TYPE]:
    """
    Get mapping from envelope names to functions that create them.

    :return:
        registry of envelopes
    """
    registry = {
        'constant': constant,
        'generic_ahdsr': generic_ahdsr,
        'relative_ahdsr': relative_ahdsr,
        'trapezoid': trapezoid,
        'user_defined': user_defined_envelope,
    }
    return registry
