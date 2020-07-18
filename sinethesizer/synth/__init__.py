"""
Synthesize sound.

Author: Nikolay Lysenko
"""


from . import effects, envelopes, timbre, timeline, utils, waves
from .effects import get_effects_registry
from .envelopes import get_envelopes_registry
from .synth import synthesize


__all__ = [
    'effects', 'envelopes', 'timbre', 'timeline', 'utils', 'waves',
    'get_effects_registry', 'get_envelopes_registry', 'synthesize'
]
