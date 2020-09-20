"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


from . import (
    equalizer,
    filter,
    filter_sweep,
    overdrive,
    registry,
    reverb,
    stereo,
    tremolo,
    vibrato
)
from .registry import EFFECT_FN_TYPE, get_effects_registry


__all__ = [
    'EFFECT_FN_TYPE',
    'equalizer',
    'filter',
    'filter_sweep',
    'get_effects_registry',
    'overdrive',
    'registry',
    'reverb',
    'stereo',
    'tremolo',
    'vibrato'
]
