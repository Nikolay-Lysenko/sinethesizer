"""
Synthesize sound.

Author: Nikolay Lysenko
"""


from . import envelopes, timbre, timeline, validation
from .envelopes import get_envelopes_registry
from .synth import synthesize


__all__ = [
    'envelopes', 'timbre', 'timeline', 'validation',
    'get_envelopes_registry', 'synthesize'
]
