"""
Map names of effects to functions applying them.

Author: Nikolay Lysenko.
"""


from typing import Callable

import numpy as np

from sinethesizer.effects.amplitude import (
    apply_amplitude_normalization, apply_compressor, apply_envelope_shaper
)
from sinethesizer.effects.automation import apply_automated_effect
from sinethesizer.effects.chorus import apply_chorus
from sinethesizer.effects.equalizer import apply_equalizer
from sinethesizer.effects.filter import apply_frequency_filter
from sinethesizer.effects.filter_sweep import apply_filter_sweep, apply_phaser
from sinethesizer.effects.overdrive import apply_overdrive
from sinethesizer.effects.reverb import apply_artificial_reverb, apply_room_reverb
from sinethesizer.effects.stereo import (
    apply_panning, apply_stereo_delay, apply_stereo_to_mono_conversion
)
from sinethesizer.effects.tremolo import apply_tremolo
from sinethesizer.effects.vibrato import apply_vibrato


EFFECT_FN_TYPE = Callable[
    [np.ndarray, 'sinethesizer.synth.core.Event'],
    np.ndarray
]


def get_effects_registry() -> dict[str, EFFECT_FN_TYPE]:
    """
    Get mapping from effect names to functions that apply effects.

    :return:
        registry of effects
    """
    registry = {
        'amplitude_normalization': apply_amplitude_normalization,
        'artificial_reverb': apply_artificial_reverb,
        'automation': apply_automated_effect,
        'chorus': apply_chorus,
        'compressor': apply_compressor,
        'envelope_shaper': apply_envelope_shaper,
        'equalizer': apply_equalizer,
        'filter': apply_frequency_filter,
        'filter_sweep': apply_filter_sweep,
        'overdrive': apply_overdrive,
        'panning': apply_panning,
        'phaser': apply_phaser,
        'room_reverb': apply_room_reverb,
        'stereo_delay': apply_stereo_delay,
        'stereo_to_mono_conversion': apply_stereo_to_mono_conversion,
        'tremolo': apply_tremolo,
        'vibrato': apply_vibrato,
    }
    return registry
