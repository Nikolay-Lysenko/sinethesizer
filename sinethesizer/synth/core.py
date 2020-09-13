"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import json
import random
from math import gcd
from typing import Dict, List, NamedTuple, Optional

import numpy as np

from sinethesizer.effects import EFFECT_FN_TYPE, get_effects_registry
from sinethesizer.envelopes import ENVELOPE_FN_TYPE
from sinethesizer.synth.event_to_amplitude_factor import (
    EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE
)
from sinethesizer.utils.waves import generate_mono_wave


class Event(NamedTuple):
    """
    Parameters of a basic audio event (loosely speaking, a played note).

    :param instrument:
        name of instrument that is used for synthesis
    :param start_time:
        start time (in seconds) of sound from beginning of its track
    :param duration:
        duration of a sound (in seconds) not including its release;
        in terms of MIDI, it is time between 'NOTE ON' and 'NOTE OFF' events
    :param frequency:
        fundamental frequency of a sound to be synthesized
    :param velocity:
        force of sound generation; it can be likened to force of piano key
        pressing; it is a float between 0 and 1; it can affect volume and
        frequency spectrum
    :param effects:
        JSON string representing list of effects that should be applied to
        resulting sound
    :param frame_rate:
        number of frames per second
    """
    instrument: str
    start_time: float
    duration: float
    frequency: float
    velocity: float
    effects: str
    frame_rate: int


class Modulator(NamedTuple):
    """
    Parameters of a wave that modulates frequency of another wave.

    :param waveform:
        form of a modulating wave
    :param frequency_ratio_numerator:
        numerator in ratio of modulating wave frequency to that of a
        modulated wave
    :param frequency_ratio_denominator:
        denominator in ratio of modulating wave frequency to that of a
        modulated wave
    :param phase:
        phase shift of a modulating wave (in radians)
    :param modulation_index_envelope_fn:
        function that takes parameters such as duration, velocity, and
        frame rate as inputs and returns amplitude envelope of a modulating
        wave (this envelope is known as modulation index envelope)
    """
    waveform: str
    frequency_ratio_numerator: int
    frequency_ratio_denominator: int
    phase: float
    modulation_index_envelope_fn: ENVELOPE_FN_TYPE


class ModulatedWave(NamedTuple):
    """
    Parameters of a wave with modulated frequency.

    :param waveform:
        form of a modulated wave (so called carrier)
    :param phase:
        phase shift of a carrier (in radians)
    :param amplitude_envelope_fn:
        function that takes parameters such as duration, velocity, and
        frame rate as inputs and returns amplitude envelope of output wave
    :param modulator:
        parameters of a modulating wave; if it is `None`, frequency is not
        modulated
    """
    waveform: str
    phase: float
    amplitude_envelope_fn: ENVELOPE_FN_TYPE
    modulator: Optional[Modulator]


def generate_modulated_wave(
        wave: ModulatedWave, frequency: float, event: Event
) -> np.ndarray:
    """
    Generate wave with modulated frequency.

    :param wave:
        parameters of the wave to be generated
    :param frequency:
        fundamental frequency of a wave to be generated (in Hz)
    :param event:
        parameters of sound event for which this function is called
    :return:
        wave with modulated frequency
    """
    if wave.modulator is None:
        modulator = None
        carrier_frequency = frequency
    else:
        numerator = wave.modulator.frequency_ratio_numerator
        denominator = wave.modulator.frequency_ratio_denominator
        divisor = gcd(numerator, denominator)
        carrier_frequency = (denominator / divisor) * frequency
        modulator_frequency = (numerator / divisor) * frequency
        index_envelope = wave.modulator.modulation_index_envelope_fn(event)
        modulator = generate_mono_wave(
            wave.modulator.waveform,
            modulator_frequency,
            index_envelope,
            event.frame_rate,
            wave.modulator.phase
        )

    amplitude_envelope = wave.amplitude_envelope_fn(event)
    result = generate_mono_wave(
        wave.waveform,
        carrier_frequency,
        amplitude_envelope,
        event.frame_rate,
        wave.phase,
        modulator
    )

    result = np.vstack((result, result))  # Two channels for stereo sound.
    return result


class Partial(NamedTuple):
    """
    Parameters of a partial (fundamental or overtone).

    :param wave:
        parameters of a wave that forms the partial
    :param frequency_ratio:
        ratio of this partial's frequency to fundamental frequency
    :param amplitude_ratio:
        declared ratio of this partial's peak amplitude to peak amplitude
        of the fundamental (both at maximum velocity); actual amplitude ratio
        may be different if effects applied to the partial and to the
        fundamental are not the same
    :param event_to_amplitude_factor_fn:
        function that maps event to its multiplicative contribution to
        partial's amplitude
    :param detuning_to_amplitude:
        mapping from a detuning size in semitones to amplitude of a wave
        with the corresponding detuned frequency; sum of slightly detuned
        waves sounds less artificial than one pure wave
    :param random_detuning_range:
        range of additional random detuning in semitones
    :param effects:
        sound effects that should be applied to this partial
    """
    wave: ModulatedWave
    frequency_ratio: float
    amplitude_ratio: float
    event_to_amplitude_factor_fn: EVENT_TO_AMPLITUDE_FACTOR_FN_TYPE
    detuning_to_amplitude: Dict[float, float]
    random_detuning_range: float
    effects: List[EFFECT_FN_TYPE]


def sum_two_sounds(
        first_sound: np.ndarray, second_sound: np.ndarray
) -> np.ndarray:
    """
    Sum two sound of probably unequal durations.

    :param first_sound:
        first sound as array of shape (n_channels, n_frames)
    :param second_sound:
        second sound as array of shape (n_channels, n_frames)
    :return:
        sum of the sounds
    """
    first_n_frames = first_sound.shape[1]
    second_n_frames = second_sound.shape[1]
    n_extra_frames = abs(first_n_frames - second_n_frames)
    padding = np.zeros((first_sound.shape[0], n_extra_frames))
    if first_n_frames > second_n_frames:
        second_sound = np.hstack((second_sound, padding))
    elif first_n_frames < second_n_frames:
        first_sound = np.hstack((first_sound, padding))
    return first_sound + second_sound


def generate_partial(partial: Partial, event: Event) -> np.ndarray:
    """
    Generate partial (fundamental or overtone).

    :param partial:
        parameters of the partial
    :param event:
        parameters of sound event for which this function is called
    :return:
        partial
    """
    sound = np.array([[], []], dtype=np.float64)
    partial_frequency = partial.frequency_ratio * event.frequency
    borders_of_random_detuning = (
        -partial.random_detuning_range / 2,
        partial.random_detuning_range / 2
    )
    params = partial.detuning_to_amplitude.items()
    for freq_shift_in_semitones, amplitude_ratio in params:
        freq_shift_in_semitones += random.uniform(*borders_of_random_detuning)
        frequency_ratio = 2 ** (freq_shift_in_semitones / 12)
        detuned_frequency = frequency_ratio * partial_frequency
        wave = generate_modulated_wave(partial.wave, detuned_frequency, event)
        wave *= amplitude_ratio
        sound = sum_two_sounds(sound, wave)
    sound *= partial.amplitude_ratio
    sound *= partial.event_to_amplitude_factor_fn(event)
    for effect_fn in partial.effects:
        sound = effect_fn(sound, event)
    return sound


class Instrument(NamedTuple):
    """
    Parameters of a virtual musical instrument.

    :param partials:
        parameters of partials
    :param amplitude_scaling:
        amplitude factor selected to prevent clipping by playing devices;
        set it to be less than inverse of sum of `amplitude_ratio` parameters
        over all partials (and, if applicable, take into account an increase in
        partials' amplitudes due to their effects)
    :param effects:
        sound effects that should be applied to outputs of the instrument
    """
    partials: List[Partial]
    amplitude_scaling: float
    effects: List[EFFECT_FN_TYPE]


def apply_event_level_effects(sound: np.ndarray, event: Event) -> np.ndarray:
    """
    Apply sound effects that are specific to a particular event.

    :param sound:
        sound to be modified
    :param event:
        event for which `sound` has been produced
    :return:
        modified sound
    """
    if not event.effects:
        return sound
    effects_registry = get_effects_registry()
    effects = json.loads(event.effects)
    for effect in effects:
        effect_name = effect.pop('name')
        sound = effects_registry[effect_name](sound, event, **effect)
    return sound


def synthesize(
        event: Event, instruments_registry: Dict[str, Instrument]
) -> np.ndarray:
    """
    Synthesize one sound event (loosely speaking, a played note).

    :param event:
        parameters of sound event to be synthesized
    :param instruments_registry:
        mapping from instrument names to their representations
    :return:
        synthesized sound as pressure deviation timeline
    """
    sound = np.array([[], []], dtype=np.float64)
    instrument = instruments_registry[event.instrument]
    for partial in instrument.partials:
        partial_sound = generate_partial(partial, event)
        sound = sum_two_sounds(sound, partial_sound)
    for effect_fn in instrument.effects:
        sound = effect_fn(sound, event)
    sound *= instrument.amplitude_scaling
    apply_event_level_effects(sound, event)
    return sound
