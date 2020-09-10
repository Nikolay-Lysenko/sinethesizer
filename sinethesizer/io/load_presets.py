"""
Create virtual instruments based on YAML file.

Author: Nikolay Lysenko
"""


import functools
from typing import List, Dict, Any

import yaml

from sinethesizer.effects import EFFECT_FN_TYPE, get_effects_registry
from sinethesizer.envelopes import ENVELOPE_FN_TYPE, get_envelopes_registry
from sinethesizer.synth.core import Instrument, ModulatedWave, Partial
from sinethesizer.synth.event_to_amplitude_factor import (
    EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE,
    get_event_to_amplitude_factor_functions_registry
)


def create_envelope_fn(envelope_data: Dict[str, Any]) -> ENVELOPE_FN_TYPE:
    """
    Create function that maps a sound event to envelope.

    :param envelope_data:
        envelope parameters
    :return:
        envelope function
    """
    envelopes_registry = get_envelopes_registry()
    envelope_fn = functools.partial(
        envelopes_registry[envelope_data['name']],
        **{k: v for k, v in envelope_data.items() if k != 'name'}
    )
    return envelope_fn


def create_list_of_effect_fns(
        effects_data: List[Dict[str, Any]]
) -> List[EFFECT_FN_TYPE]:
    """
    Create list of functions that apply sound effects to timelines.

    :param effects_data:
        effects parameters
    :return:
        sound effects functions
    """
    effects_registry = get_effects_registry()
    effects_fns = []
    for effect_data in effects_data:
        effect_fn = functools.partial(
            effects_registry[effect_data['name']],
            **{k: v for k, v in effect_data.items() if k != 'name'}
        )
        effects_fns.append(effect_fn)
    return effects_fns


def create_event_to_amplitude_factor_fn(
        fn_data: Dict[str, Any]
) -> EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE:
    """
    Create function for multiplicative contribution of event to amplitude.

    :param fn_data:
        function parameters
    :return:
        amplitude factor function
    """
    registry = get_event_to_amplitude_factor_functions_registry()
    event_to_amplitude_factor_fn = functools.partial(
        registry[fn_data['name']],
        **{k: v for k, v in fn_data.items() if k != 'name'}
    )
    return event_to_amplitude_factor_fn


def norm_amplitudes_of_detuned_waves(
        detuning_to_amplitude: Dict[float, float]
) -> Dict[float, float]:
    """
    Norm amplitudes of detuned waves to sum up to 1.

    :param detuning_to_amplitude:
        mapping from a detuning size in semitones to relative amplitude
        of a wave with the corresponding detuned frequency
    :return:
        input mapping modified to have unit sum of values
    """
    denominator = sum(v for k, v in detuning_to_amplitude.items())
    return {k: v / denominator for k, v in detuning_to_amplitude.items()}


def convert_modulated_wave(wave_data: Dict[str, Any]) -> ModulatedWave:
    """
    Convert representation of modulated wave to internal data structure.

    :param wave_data:
        parameters of modulated wave as dictionary
    :return:
        parameters of modulated wave as internal data structure
    """
    modulated_wave = ModulatedWave(
        amplitude_envelope_fn=create_envelope_fn(
            wave_data['amplitude_envelope_fn']
        ),
        carrier_waveform=wave_data['carrier_waveform'],
        carrier_phase=wave_data.get('carrier_phase', 0),
        modulation_index_envelope_fn=create_envelope_fn(
            wave_data['modulation_index_envelope_fn']
        ),
        modulator_waveform=wave_data['modulator_waveform'],
        modulator_frequency_ratio=wave_data['modulator_frequency_ratio'],
        modulator_phase=wave_data.get('modulator_phase', 0)
    )
    return modulated_wave


def convert_partials(partials_data: List[Dict[str, Any]]) -> List[Partial]:
    """
    Convert representations of partials to internal data structures.

    :param partials_data:
        parameters of partials as dictionaries
    :return:
        parameters of partials as internal data structures
    """
    partials = []
    for partial_data in partials_data:
        partial = Partial(
            wave=convert_modulated_wave(partial_data['wave']),
            frequency_ratio=partial_data['frequency_ratio'],
            amplitude_ratio=partial_data['amplitude_ratio'],
            event_to_amplitude_factor_fn=create_event_to_amplitude_factor_fn(
                partial_data['event_to_amplitude_factor_fn']
            ),
            detuning_to_amplitude=norm_amplitudes_of_detuned_waves(
                partial_data['detuning_to_amplitude']
            ),
            random_detuning_range=partial_data['random_detuning_range'],
            effects=create_list_of_effect_fns(partial_data.get('effects', []))
        )
        partials.append(partial)
    return partials


def create_instruments_registry(input_path: str) -> Dict[str, Any]:
    """
    Create mapping from instrument names to their representations.

    :param input_path:
        path to YAML file with definitions of instruments
    :return:
        instruments registry
    """
    with open(input_path) as input_file:
        input_data = yaml.safe_load(input_file)
    instruments_registry = {}
    for instrument_data in input_data:
        instruments_registry[instrument_data['name']] = Instrument(
            partials=convert_partials(instrument_data['partials']),
            amplitude_factor=instrument_data['amplitude_factor'],
            effects=create_list_of_effect_fns(
                instrument_data.get('effects', [])
            )
        )
    return instruments_registry
