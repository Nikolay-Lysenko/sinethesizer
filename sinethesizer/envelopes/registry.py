"""
Map names of envelopes to functions applying them.

Author: Nikolay Lysenko
"""


from typing import Callable, Dict

import numpy as np

from sinethesizer.envelopes.ahdsr import generic_ahdsr
from sinethesizer.envelopes.user_defined import user_defined_envelope


ENVELOPE_FN_TYPE = Callable[['sinethesizer.synth.core.Task'], np.ndarray]


def get_envelopes_registry() -> Dict[str, ENVELOPE_FN_TYPE]:
    """
    Get mapping from envelope names to functions that create them.

    :return:
        registry of envelopes
    """
    registry = {
        'generic_ahdsr': generic_ahdsr,
        'user_defined': user_defined_envelope,
    }
    return registry