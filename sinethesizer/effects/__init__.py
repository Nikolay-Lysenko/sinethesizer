"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


from . import (
    filter, filter_sweep, overdrive, registry, reverb, tremolo, vibrato
)
from .registry import EFFECT_FN_TYPE, get_effects_registry


__all__ = [
    'EFFECT_FN_TYPE',
    'filter',
    'filter_sweep',
    'get_effects_registry',
    'overdrive',
    'registry',
    'reverb',
    'tremolo',
    'vibrato'
]
