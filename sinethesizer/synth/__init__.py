"""
Synthesize sound.

Author: Nikolay Lysenko
"""


from . import envelopes, effects, timbre, utils, waves
from .envelopes import get_envelopes_registry
from .effects import get_effects_registry
from .synth import synthesize


__all__ = [
    'envelopes.py', 'effects', 'timbre', 'utils', 'waves',
    'get_effects_registry', 'get_envelopes_registry', 'synthesize'
]
