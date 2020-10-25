"""
Apply sound effects with settings changing over time.

In particular, this module can be used to apply equalizer/filter envelope.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List

import numpy as np

from sinethesizer.effects.equalizer import apply_equalizer
from sinethesizer.effects.filter import apply_frequency_filter
from sinethesizer.effects.filter_sweep import apply_filter_sweep, apply_phaser
from sinethesizer.effects.overdrive import apply_overdrive
from sinethesizer.effects.reverb import apply_reverb
from sinethesizer.effects.stereo import apply_haas_effect, apply_panning
from sinethesizer.effects.tremolo import apply_tremolo
from sinethesizer.effects.vibrato import apply_vibrato
from sinethesizer.effects.volume import apply_amplitude_normalization


REGISTRY_OF_AUTOMATABLE_EFFECTS = {
    'amplitude_normalization': apply_amplitude_normalization,
    'equalizer': apply_equalizer,
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


def apply_automated_effect(
        sound: np.ndarray, event: 'sinethesizer.synth.core.Event',
        automated_effect_name: str, break_points: List[Dict[str, Any]],
        **kwargs
) -> np.ndarray:
    """
    Apply an effect with settings changing over time.

    :param sound:
        sound to be modified
    :param event:
        parameters of sound event for which this function is called
    :param automated_effect_name:
        name of function from the registry that applies the effect
    :param break_points:
        points that define dynamics of effect parameters; each point must be a
        dictionary that has the key 'relative_position' with a float value
        between 0 and 1 that defines position of the point on time axis;
        also a dictionary may include key-value pairs with parameters of the
        effect; output sound is linearly interpolated at intermediate points
    :return:
        modified sound
    """
    if break_points[0]['relative_position'] != 0:
        raise ValueError("Effect automation must start from 0.")
    if break_points[-1]['relative_position'] != 1:
        raise ValueError("Effect automation must end with 1.")

    n_frames = sound.shape[1]
    indices = []
    effects_params = []
    for break_point in break_points:
        index = int(round(break_point['relative_position'] * n_frames))
        indices.append(index)
        effect_params = {
            k: v for k, v in break_point.items() if k != 'relative_position'
        }
        effects_params.append(effect_params)
    indices.insert(0, indices[0])
    indices.append(indices[-1])

    processed_sound = np.zeros_like(sound)
    effect_fn = REGISTRY_OF_AUTOMATABLE_EFFECTS[automated_effect_name]
    zipped = zip(indices, indices[1:], indices[2:], effects_params)
    for start_index, center_index, end_index, effect_params in zipped:
        fragment = np.copy(sound[:, start_index:end_index])
        processed_fragment = effect_fn(
            fragment, event, **effect_params, **kwargs
        )
        if center_index - start_index > 0:
            asc_weights = np.linspace(0, 1, center_index - start_index, False)
        else:
            asc_weights = np.array([])
        if end_index - center_index > 0:
            desc_weights = np.linspace(1, 0, end_index - center_index, False)
        else:
            desc_weights = np.array([])
        weights = np.hstack((asc_weights, desc_weights))

        processed_fragment *= weights
        processed_sound[:, start_index:end_index] += processed_fragment
    return processed_sound
