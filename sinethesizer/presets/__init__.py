"""
Define timbre parameters of some virtual instruments.

Author: Nikolay Lysenko
"""


from . import adsr_envelopes, basic_timbres
from .registry import TIMBRES_REGISTRY


__all__ = ['adsr_envelopes', 'basic_timbres', 'TIMBRES_REGISTRY']
