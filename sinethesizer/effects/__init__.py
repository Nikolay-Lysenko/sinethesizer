"""
Modify sound with effects.

Author: Nikolay Lysenko
"""


from . import (
    automation,
    equalizer,
    filter,
    filter_sweep,
    overdrive,
    registry,
    reverb,
    stereo,
    tremolo,
    vibrato,
    volume,
)
from .registry import EFFECT_FN_TYPE, get_effects_registry


__all__ = [
    'EFFECT_FN_TYPE',
    'automation',
    'equalizer',
    'filter',
    'filter_sweep',
    'get_effects_registry',
    'overdrive',
    'registry',
    'reverb',
    'stereo',
    'tremolo',
    'vibrato',
    'volume',
]
