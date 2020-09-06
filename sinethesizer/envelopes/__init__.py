"""
Define envelopes for wave amplitude and modulation index.

Author: Nikolay Lysenko
"""


from . import ahdsr, user_defined
from .registry import ENVELOPE_FN_TYPE, get_envelopes_registry


__all__ = [
    'ENVELOPE_FN_TYPE',
    'ahdsr',
    'get_envelopes_registry',
    'user_defined'
]
