"""
Map names of effects to functions applying them.

Author: Nikolay Lysenko.
"""


from typing import Any, Callable, Dict

import numpy as np

from sinethesizer.effects.filter import apply_frequency_filter
from sinethesizer.effects.filter_sweep import (
    apply_filter_sweep, apply_phaser
)
from sinethesizer.effects.overdrive import apply_overdrive
from sinethesizer.effects.reverb import apply_reverb
from sinethesizer.effects.tremolo import apply_tremolo
from sinethesizer.effects.vibrato import apply_vibrato


EFFECT_FN_TYPE = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]


def get_effects_registry() -> Dict[str, EFFECT_FN_TYPE]:
    """
    Get mapping from effect names to functions that apply effects.

    :return:
        registry of effects
    """
    registry = {
        'filter': apply_frequency_filter,
        'filter_sweep': apply_filter_sweep,
        'overdrive': apply_overdrive,
        'phaser': apply_phaser,
        'reverb': apply_reverb,
        'tremolo': apply_tremolo,
        'vibrato': apply_vibrato,
    }
    return registry
