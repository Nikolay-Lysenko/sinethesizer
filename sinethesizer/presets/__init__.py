"""
Define timbre parameters of some virtual instruments.

Author: Nikolay Lysenko
"""


from . import adsr_envelopes, effects, timbres
from .registry import EFFECTS_REGISTRY, TIMBRES_REGISTRY


__all__ = ['adsr_envelopes', 'timbres', 'EFFECTS_REGISTRY', 'TIMBRES_REGISTRY']
