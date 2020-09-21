"""
Map names of effects to functions applying them.

Author: Nikolay Lysenko.
"""


from typing import Callable, Dict

import numpy as np

from sinethesizer.effects.equalizer import (
    apply_equalizer, apply_equalizer_envelope
)
from sinethesizer.effects.filter import apply_frequency_filter
from sinethesizer.effects.filter_sweep import apply_filter_sweep, apply_phaser
from sinethesizer.effects.overdrive import apply_overdrive
from sinethesizer.effects.reverb import apply_reverb
from sinethesizer.effects.stereo import apply_haas_effect, apply_panning
from sinethesizer.effects.tremolo import apply_tremolo
from sinethesizer.effects.vibrato import apply_vibrato


EFFECT_FN_TYPE = Callable[
    [np.ndarray, 'sinethesizer.synth.core.Event'],
    np.ndarray
]


def get_effects_registry() -> Dict[str, EFFECT_FN_TYPE]:
    """
    Get mapping from effect names to functions that apply effects.

    :return:
        registry of effects
    """
    registry = {
        'equalizer': apply_equalizer,
        'equalizer_envelope': apply_equalizer_envelope,
        'filter': apply_frequency_filter,
        'filter_sweep': apply_filter_sweep,
        'haas': apply_haas_effect,
        'overdrive': apply_overdrive,
        'panning': apply_panning,
        'phaser': apply_phaser,
        'reverb': apply_reverb,
        'tremolo': apply_tremolo,
        'vibrato': apply_vibrato,
    }
    return registry
