"""
Synthesize sound.

Author: Nikolay Lysenko
"""


from . import envelopes, timbre, timeline, utils
from .envelopes import get_envelopes_registry
from .synth import synthesize


__all__ = [
    'envelopes', 'timbre', 'timeline', 'utils',
    'get_envelopes_registry', 'synthesize'
]
