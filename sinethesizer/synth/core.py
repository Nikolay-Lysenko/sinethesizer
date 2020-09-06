"""
Synthesize sound.

Author: Nikolay Lysenko
"""


import random
from typing import Dict, List, NamedTuple

import numpy as np

from sinethesizer.effects import EFFECT_FN_TYPE
from sinethesizer.envelopes import ENVELOPE_FN_TYPE
from sinethesizer.synth.partials_amplitude import PARTIALS_AMPLITUDE_FN_TYPE
from sinethesizer.utils.waves import NAME_TO_WAVEFORM


class Task(NamedTuple):
    """
    Specifications of basic sound synthesis task.

    :param duration:
        duration of an event (in seconds) not including its release;
        in terms of MIDI, it is time between 'NOTE ON' and 'NOTE OFF' events
    :param frequency:
        fundamental frequency of a sound to be synthesized
    :param volume:
        ratio of the highest amplitude of the resulting sound to maximum
        amplitude that is not clipped by playing devices; it is a float
        between 0 and 1
    :param velocity:
        force of sound generation; it can be likened to force of piano key
        pressing; it is a float between 0 and 1
    :param frame_rate:
        number of frames per second
    :param effects:
        list of effects that should be applied to resulting sound
    """
    duration: float
    frequency: float
    volume: float
    velocity: float
    frame_rate: int
    effects: List[EFFECT_FN_TYPE]


class ModulatedWave(NamedTuple):
    """
    Specifications of a wave with modulated frequency.

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
        wave: ModulatedWave, carrier_frequency: float, task: Task
) -> np.ndarray:
    """
    Generate wave with modulated frequency.

    :param wave:
        specifications of the wave to be generated
    :param carrier_frequency:
        frequency of a carrier wave (in Hz); loosely speaking,
        it is a base frequency of the wave to be generated
    :param task:
        parameters of sound synthesis task that triggered generation
        of the wave
    :return:
        wave with modulated frequency
    """
    amplitude_envelope = wave.amplitude_envelope_fn(task)
    duration_in_frames = len(amplitude_envelope)
    frame_rate = task.frame_rate

    mod_frequency = wave.modulator_frequency_ratio * carrier_frequency
    mod_period_in_frames = frame_rate / mod_frequency
    mod_phase_in_frames = mod_period_in_frames * wave.modulator_phase
    mod_phase_in_frames = int(round(mod_phase_in_frames))
    mod_xs = np.arange(duration_in_frames) + mod_phase_in_frames

    mod_wave_fn = NAME_TO_WAVEFORM[wave.modulator_waveform]
    modulator = mod_wave_fn(2 * np.pi * mod_frequency / frame_rate * mod_xs)
    modulation_index_envelope = wave.modulation_index_envelope_fn(task)
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
    Specifications of a partial (fundamental or overtone).

    :param wave:
        specifications of a wave that forms the partial
    :param frequency_ratio:
        ratio of this partial's frequency to fundamental frequency
    :param detuning_to_amplitude:
        mapping from a detuning size in semitones to relative amplitude of
        a wave with the corresponding detuned frequency; sum of slightly
        detuned waves sounds less artificial than one pure wave
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


def generate_partial(partial: Partial, task: Task) -> np.ndarray:
    """
    Generate partial (fundamental or overtone).

    :param partial:
        specifications of the partial
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
    Specifications of a virtual musical instrument.

    :param partials:
        specifications of partials
    :param partials_amplitude_fn:
        function that takes parameters such as position of a partial and
        velocity as inputs and returns ratio of the partial's amplitude to
        that of the fundamental
    :param effects:
        sound effects that should be applied to outputs of the instrument
    """
    partials: List[Partial]
    partials_amplitude_fn: PARTIALS_AMPLITUDE_FN_TYPE
    effects: List[EFFECT_FN_TYPE]


def synthesize(instrument: Instrument, task: Task) -> np.ndarray:
    """
    Synthesize one sound event (loosely speaking, a played note).

    :param instrument:
        specifications of a virtual musical instrument
    :param task:
        specification of sound event to be synthesized
    :return:
        synthesized sound as pressure deviation timeline
    """
    sound = np.array([[], []], dtype=np.float64)
    partials_amplitude_fn = instrument.partials_amplitude_fn
    for partial_id, partial in enumerate(instrument.partials):
        partial_sound = generate_partial(partial, task)
        partial_sound *= partials_amplitude_fn(partial_id, task)
        sound = sum_two_sounds(sound, partial_sound)

    for effect_fn in instrument.effects:
        sound = effect_fn(sound, task)
    sound *= task.volume / np.max(np.abs(sound))
    for effect_fn in task.effects:
        sound = effect_fn(sound, task)

    return sound
