"""
Synthesize sound.

Author: Nikolay Lysenko
"""


from . import adsr_envelopes, effects, timbre, utils, waves
from .adsr_envelopes import get_envelopes_registry
from .effects import get_effects_registry
from .synth import synthesize


__all__ = [
    'adsr_envelopes', 'effects', 'timbre', 'utils', 'waves',
    'get_effects_registry', 'get_envelopes_registry', 'synthesize'
]
