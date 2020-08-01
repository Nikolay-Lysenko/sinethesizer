"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


from . import filter, filter_sweep, overdrive, registry
from .registry import EFFECT_FN_TYPE, get_effects_registry


__all__ = [
    'EFFECT_FN_TYPE',
    'filter',
    'get_effects_registry',
    'overdrive',
    'registry'
]
