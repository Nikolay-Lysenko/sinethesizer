"""
Create timbres of virtual instruments based on YAML file.

Author: Nikolay Lysenko
"""


from functools import partial
from typing import List, Dict, Any

import yaml

from sinethesizer.synth import get_effects_registry, get_envelopes_registry
from sinethesizer.synth.adsr_envelopes import ENVELOPE_FN_TYPE
from sinethesizer.synth.effects import EFFECT_FN_TYPE
from sinethesizer.synth.timbre import OvertoneSpec, TimbreSpec


def create_volume_envelope_fn(
        envelope_data: Dict[str, Any]
) -> ENVELOPE_FN_TYPE:
    """
    Create function that maps duration and frame rate to volume envelope.

    :param envelope_data:
        envelope parameters
    :return:
        volume envelope function
    """
    envelopes_registry = get_envelopes_registry()
    volume_envelope_fn = partial(
        envelopes_registry[envelope_data['name']],
        **{k: v for k, v in envelope_data.items() if k != 'name'}
    )
    return volume_envelope_fn


def create_list_of_effect_fns(
        effects_data: List[Dict[str, Any]]
) -> List[EFFECT_FN_TYPE]:
    """
    Create list of functions that apply sound effects to timeline.

    :param effects_data:
        effects parameters
    :return:
        sound effects functions
    """
    effects_registry = get_effects_registry()
    effects_fns = []
    for effect_data in effects_data:
        effect_fn = partial(
            effects_registry[effect_data['name']],
            **{k: v for k, v in effect_data.items() if k != 'name'}
        )
        effects_fns.append(effect_fn)
    return effects_fns


def create_overtones_specs(
        overtones_data: List[Dict[str, Any]]
) -> List[OvertoneSpec]:
    """
    Convert overtone specifications to internal data structure.

    :param overtones_data:
        parameters of overtones
    :return:
        specifications of overtones
    """
    overtones_specs = []
    for overtone_data in overtones_data:
        overtone_spec = OvertoneSpec(
            waveform=overtone_data['waveform'],
            frequency_ratio=overtone_data['frequency_ratio'],
            volume_share=overtone_data['volume_share'],
            volume_envelope_fn=create_volume_envelope_fn(
                overtone_data['volume_envelope']
            ),
            effects=create_list_of_effect_fns(
                overtone_data.get('effects', [])
            )
        )
        overtones_specs.append(overtone_spec)
    return overtones_specs


def create_timbres_registry(input_path: str) -> Dict[str, Any]:
    """
    Create mapping from timbre names to their specifications.

    :param input_path:
        path to YAML file with definitions of timbres
    :return:
        timbres registry
    """
    with open(input_path) as input_file:
        input_data = yaml.safe_load(input_file)
    timbres_registry = {}
    for timbre_data in input_data:
        timbres_registry[timbre_data['name']] = TimbreSpec(
            fundamental_waveform=timbre_data['fundamental_waveform'],
            fundamental_volume_envelope_fn=create_volume_envelope_fn(
                timbre_data['fundamental_volume_envelope']
            ),
            fundamental_effects=create_list_of_effect_fns(
                timbre_data.get('fundamental_effects', [])
            ),
            overtones_specs=create_overtones_specs(
                timbre_data.get('overtones_specs', [])
            )
        )
    return timbres_registry
