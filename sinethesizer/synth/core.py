"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import json
import random
from typing import Dict, List, NamedTuple

import numpy as np

from sinethesizer.effects import EFFECT_FN_TYPE, get_effects_registry
from sinethesizer.envelopes import ENVELOPE_FN_TYPE
from sinethesizer.synth.partials_amplitude import PARTIALS_AMPLITUDE_FN_TYPE
from sinethesizer.utils.waves import NAME_TO_WAVEFORM


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
        pressing; it is a float between 0 and 1; it affects volume and,
        maybe, frequency spectrum
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


class ModulatedWave(NamedTuple):
    """
    Parameters of a wave with modulated frequency.

    :param amplitude_envelope_fn:
        function that takes parameters such as duration, velocity, and
        frame rate as inputs and returns amplitude envelope of output wave
    :param carrier_waveform:
        form of a modulated wave (so called carrier)
    :param carrier_phase:
        phase shift of a carrier as a fraction of its period;
        a float between 0 and 1
    :param modulation_index_envelope_fn:
        function that takes parameters such as duration, velocity, and
        frame rate as inputs and returns amplitude envelope of a modulating
        wave (this envelope is known as modulation index envelope)
    :param modulator_waveform:
        form of a modulating wave (so called modulator)
    :param modulator_frequency_ratio:
        ratio of modulator frequency to that of a carrier
    :param modulator_phase:
        phase shift of a modulator as a fraction of its period;
        a float between 0 and 1
    """
    amplitude_envelope_fn: ENVELOPE_FN_TYPE
    carrier_waveform: str
    carrier_phase: float
    modulation_index_envelope_fn: ENVELOPE_FN_TYPE
    modulator_waveform: str
    modulator_frequency_ratio: float
    modulator_phase: float


def generate_modulated_wave(
        wave: ModulatedWave, carrier_frequency: float, event: Event
) -> np.ndarray:
    """
    Generate wave with modulated frequency.

    :param wave:
        parameters of the wave to be generated
    :param carrier_frequency:
        frequency of a carrier wave (in Hz); loosely speaking,
        it is a base frequency of the wave to be generated
    :param event:
        parameters of sound synthesis task that triggered generation
        of the wave
    :return:
        wave with modulated frequency
    """
    amplitude_envelope = wave.amplitude_envelope_fn(event)
    duration_in_frames = len(amplitude_envelope)
    frame_rate = event.frame_rate

    mod_frequency = wave.modulator_frequency_ratio * carrier_frequency
    mod_period_in_frames = frame_rate / mod_frequency
    mod_phase_in_frames = mod_period_in_frames * wave.modulator_phase
    mod_phase_in_frames = int(round(mod_phase_in_frames))
    mod_xs = np.arange(duration_in_frames) + mod_phase_in_frames

    mod_wave_fn = NAME_TO_WAVEFORM[wave.modulator_waveform]
    modulator = mod_wave_fn(2 * np.pi * mod_frequency / frame_rate * mod_xs)
    modulation_index_envelope = wave.modulation_index_envelope_fn(event)
    modulator *= modulation_index_envelope

    carr_period_in_frames = frame_rate / carrier_frequency
    carr_phase_in_frames = carr_period_in_frames * wave.carrier_phase
    carr_phase_in_frames = int(round(carr_phase_in_frames))
    carr_xs = np.arange(duration_in_frames) + carr_phase_in_frames
    xs = carr_xs + modulator

    carr_wave_fn = NAME_TO_WAVEFORM[wave.carrier_waveform]
    result = carr_wave_fn(2 * np.pi * carrier_frequency / frame_rate * xs)
    result *= amplitude_envelope

    result = np.vstack((result, result))  # Two channels for stereo sound.
    return result


class Partial(NamedTuple):
    """
    Parameters of a partial (fundamental or overtone).

    :param wave:
        parameters of a wave that forms the partial
    :param frequency_ratio:
        ratio of this partial's frequency to fundamental frequency
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


def generate_partial(partial: Partial, task: Event) -> np.ndarray:
    """
    Generate partial (fundamental or overtone).

    :param partial:
        parameters of the partial
    :param task:
        parameters of sound synthesis task that triggered generation
        of the partial
    :return:
        partial
    """
    sound = np.array([[], []], dtype=np.float64)
    partial_frequency = partial.frequency_ratio * task.frequency
    borders_of_random_detuning = (
        -partial.random_detuning_range / 2,
        partial.random_detuning_range / 2
    )
    params = partial.detuning_to_amplitude.items()
    for freq_shift_in_semitones, amplitude_ratio in params:
        freq_shift_in_semitones += random.uniform(*borders_of_random_detuning)
        frequency_ratio = 2 ** (freq_shift_in_semitones / 12)
        detuned_frequency = frequency_ratio * partial_frequency
        wave = generate_modulated_wave(partial.wave, detuned_frequency, task)
        wave *= amplitude_ratio
        sound = sum_two_sounds(sound, wave)
    for effect_fn in partial.effects:
        sound = effect_fn(sound, task)
    return sound


class Instrument(NamedTuple):
    """
    Parameters of a virtual musical instrument.

    :param partials:
        parameters of partials
    :param partials_amplitude_fn:
        function that takes parameters such as position of a partial and
        velocity as inputs and returns ratio of the partial's amplitude to
        that of the fundamental
    :param amplitude_factor:
        amplitude factor selected to prevent clipping by playing devices
    :param effects:
        sound effects that should be applied to outputs of the instrument
    """
    partials: List[Partial]
    partials_amplitude_fn: PARTIALS_AMPLITUDE_FN_TYPE
    amplitude_factor: float
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
    partials_amplitude_fn = instrument.partials_amplitude_fn
    for partial_id, partial in enumerate(instrument.partials):
        partial_sound = generate_partial(partial, event)
        partial_sound *= partials_amplitude_fn(partial_id, event)
        sound = sum_two_sounds(sound, partial_sound)

    for effect_fn in instrument.effects:
        sound = effect_fn(sound, event)
    apply_event_level_effects(sound, event)
    sound *= instrument.amplitude_factor
    return sound
