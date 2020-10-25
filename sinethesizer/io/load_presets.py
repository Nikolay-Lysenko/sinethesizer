"""
Create virtual instruments based on YAML file.

Author: Nikolay Lysenko
"""


import functools
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import yaml

from sinethesizer.effects import EFFECT_FN_TYPE, get_effects_registry
from sinethesizer.envelopes import ENVELOPE_FN_TYPE, get_envelopes_registry
from sinethesizer.synth.core import (
    Instrument, ModulatedWave, Modulator, Partial
)
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


def convert_modulator(
        modulator_data: Optional[Dict[str, Any]]
) -> Optional[Modulator]:
    """
    Convert representation of modulating wave to internal data structure.

    :param modulator_data:
        parameters of modulating wave as dictionary
    :return:
        parameters of modulating wave as internal data structure
    """
    if modulator_data is None:
        return None
    modulator = Modulator(
        waveform=modulator_data['waveform'],
        carrier_frequency_ratio=modulator_data['carrier_frequency_ratio'],
        modulator_frequency_ratio=modulator_data['modulator_frequency_ratio'],
        modulation_index_envelope_fn=create_envelope_fn(
            modulator_data['modulation_index_envelope_fn']
        ),
        phase=modulator_data.get('phase', 0),
        use_ring_modulation=modulator_data.get('use_ring_modulation', False)
    )
    return modulator


def compute_frequency_ratios(
        modulator_data: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Compute ratios of carrier and modulator frequencies to that of fundamental.

    :param modulator_data:
        parameters of modulating wave as dictionary
    :return:
        ratios of carrier and modulator frequencies to that of fundamental
    """
    numerator = modulator_data['frequency_ratio_numerator']
    denominator = modulator_data['frequency_ratio_denominator']
    divisor = math.gcd(numerator, denominator)
    frequency_scaling = max(denominator, numerator) / divisor
    # If above value is too high, scaled frequencies are above Nyquist
    # frequency and so output is bad. Fortunately, absence of scaling
    # works good in this case, because missing fundamental lies below
    # audible range.
    # However, if above value is neither low nor too high, there is no
    # good option, because for high notes missing fundamental lies inside
    # audible range, but scaled frequencies are above Nyquist frequency.
    # So let us choose thresholds.
    # If `frequency_scaling` <= 10, notes up to C7 are played without aliasing
    # given frame rate equal to 44100.
    # If `frequency_scaling` >= 50, missing fundamentals for notes up to C6
    # are in infrasonic range.
    min_threshold = 10
    max_threshold = 50
    if frequency_scaling <= min_threshold:
        carrier_frequency_ratio = denominator / divisor
        modulator_frequency_ratio = numerator / divisor
    else:
        carrier_frequency_ratio = 1
        modulator_frequency_ratio = numerator / denominator
        if frequency_scaling <= max_threshold:  # pragma: no cover
            warnings.warn(
                'Problems with missing fundamental may happen.',
                UserWarning
            )
    return carrier_frequency_ratio, modulator_frequency_ratio


def convert_modulated_wave(wave_data: Dict[str, Any]) -> ModulatedWave:
    """
    Convert representation of modulated wave to internal data structure.

    :param wave_data:
        parameters of modulated wave as dictionary
    :return:
        parameters of modulated wave as internal data structure
    """
    # Output wave of PM acts like carrier wave for AM. So frequency ratios for
    # PM must be multiplied by carrier frequency ratio for AM (if it exists).
    factor = 1
    keys = ['amplitude_modulator', 'phase_modulator']
    am_enabled = 'amplitude_modulator' in wave_data.keys()
    pm_enabled = 'phase_modulator' in wave_data.keys()
    enabledness = [am_enabled, pm_enabled]
    for key, enabled in zip(keys, enabledness):
        if not enabled:
            continue
        carrier_frequency_ratio, modulator_frequency_ratio = (
            compute_frequency_ratios(wave_data[key])
        )
        carrier_frequency_ratio *= factor
        modulator_frequency_ratio *= factor
        factor = carrier_frequency_ratio
        wave_data[key]['carrier_frequency_ratio'] = carrier_frequency_ratio
        wave_data[key]['modulator_frequency_ratio'] = modulator_frequency_ratio

    modulated_wave = ModulatedWave(
        waveform=wave_data['waveform'],
        amplitude_envelope_fn=create_envelope_fn(
            wave_data['amplitude_envelope_fn']
        ),
        phase=wave_data.get('phase', 0),
        amplitude_modulator=convert_modulator(
            wave_data.get('amplitude_modulator')
        ),
        phase_modulator=convert_modulator(
            wave_data.get('phase_modulator')
        ),
        quasiperiodic_bandwidth=wave_data.get('quasiperiodic_bandwidth', 0)
    )
    return modulated_wave


def norm_amplitudes_of_detuned_waves(
        detuning_to_amplitude: Dict[float, float]
) -> Dict[float, float]:
    """
    Norm amplitudes of detuned waves to sum up to 1.

    :param detuning_to_amplitude:
        mapping from detuning size in semitones to relative amplitude
        of a wave with the corresponding detuned frequency
    :return:
        input mapping modified to have unit sum of values
    """
    denominator = sum(v for k, v in detuning_to_amplitude.items())
    return {k: v / denominator for k, v in detuning_to_amplitude.items()}


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
    if fn_data is None:
        fn_data = {'name': 'power_fn_of_velocity', 'power': 1.0}
    registry = get_event_to_amplitude_factor_functions_registry()
    event_to_amplitude_factor_fn = functools.partial(
        registry[fn_data['name']],
        **{k: v for k, v in fn_data.items() if k != 'name'}
    )
    return event_to_amplitude_factor_fn


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
                partial_data.get('event_to_amplitude_factor_fn')
            ),
            random_detuning_range=partial_data.get('random_detuning_range', 0),
            detuning_to_amplitude=norm_amplitudes_of_detuned_waves(
                partial_data.get('detuning_to_amplitude', {0: 1})
            ),
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
            amplitude_scaling=instrument_data['amplitude_scaling'],
            effects=create_list_of_effect_fns(
                instrument_data.get('effects', [])
            )
        )
    return instruments_registry
