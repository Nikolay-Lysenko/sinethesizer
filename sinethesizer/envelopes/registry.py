"""
Map names of envelopes to functions creating them.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict

import numpy as np

from sinethesizer.envelopes.ahdsr import (
    create_generic_ahdsr_envelope,
    create_relative_ahdsr_envelope,
    create_trapezoid_envelope
)
from sinethesizer.envelopes.misc import (
    create_constant_envelope, create_exponentially_decaying_envelope
)
from sinethesizer.envelopes.user_defined import create_user_defined_envelope


ENVELOPE_FN_TYPE = Callable[['sinethesizer.synth.core.Event'], np.ndarray]


def get_envelopes_registry() -> Dict[str, ENVELOPE_FN_TYPE]:
    """
    Get mapping from envelope names to functions that create them.

    :return:
        registry of envelopes
    """
    registry = {
        'constant': create_constant_envelope,
        'exponentially_decaying': create_exponentially_decaying_envelope,
        'generic_ahdsr': create_generic_ahdsr_envelope,
        'relative_ahdsr': create_relative_ahdsr_envelope,
        'trapezoid': create_trapezoid_envelope,
        'user_defined': create_user_defined_envelope,
    }
    return registry
