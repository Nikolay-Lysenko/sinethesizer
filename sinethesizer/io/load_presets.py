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
from sinethesizer.synth.partials_amplitude import (
    PARTIALS_AMPLITUDE_FN_TYPE, get_partials_amplitude_functions_registry
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


def create_partials_amplitude_fn(
        partial_amplitude_fn_data: Dict[str, Any]
) -> PARTIALS_AMPLITUDE_FN_TYPE:
    """
    Create function that defines amplitudes of all partials depending on task.

    :param partial_amplitude_fn_data:
        function parameters
    :return:
        partials amplitude function
    """
    partials_ampl_fn_registry = get_partials_amplitude_functions_registry()
    partial_amplitude_fn = functools.partial(
        partials_ampl_fn_registry[partial_amplitude_fn_data['name']],
        **{k: v for k, v in partial_amplitude_fn_data.items() if k != 'name'}
    )
    return partial_amplitude_fn


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
    modulated_wave_specs = ModulatedWave(
        amplitude_envelope_fn=create_envelope_fn(
            wave_data['amplitude_envelope_fn']
        ),
        carrier_waveform=wave_data['carrier_waveform'],
        carrier_phase=wave_data['carrier_phase'],
        modulation_index_envelope_fn=create_envelope_fn(
            wave_data['modulation_index_envelope_fn']
        ),
        modulator_waveform=wave_data['modulator_waveform'],
        modulator_frequency_ratio=wave_data['modulator_frequency_ratio'],
        modulator_phase=wave_data['modulator_phase']
    )
    return modulated_wave_specs


def convert_partials(partials_data: List[Dict[str, Any]]) -> List[Partial]:
    """
    Convert representations of partials to internal data structures.

    :param partials_data:
        parameters of partials as dictionaries
    :return:
        parameters of partials as internal data structures
    """
    partials_specs = []
    for partial_data in partials_data:
        partial_specs = Partial(
            wave=convert_modulated_wave(partial_data['wave']),
            frequency_ratio=partial_data['frequency_ratio'],
            detuning_to_amplitude=norm_amplitudes_of_detuned_waves(
                partial_data['detuning_to_amplitude']
            ),
            random_detuning_range=partial_data['random_detuning_range'],
            effects=create_list_of_effect_fns(partial_data.get('effects', []))
        )
        partials_specs.append(partial_specs)
    return partials_specs


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
            partials_amplitude_fn=create_partials_amplitude_fn(
                instrument_data['partials_amplitude_fn']
            ),
            amplitude_factor=instrument_data['amplitude_factor'],
            effects=create_list_of_effect_fns(
                instrument_data.get('effects', [])
            )
        )
    return instruments_registry
